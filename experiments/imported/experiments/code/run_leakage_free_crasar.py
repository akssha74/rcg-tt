#!/usr/bin/env python3
"""Leakage-free CRASAR training with guarded spatial blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_primary_multiseed as base


ROOT = Path(__file__).resolve().parents[2]
REPO = "CRASAR/CRASAR-U-DROIDs"
SNAPSHOT = "47cf4ab3a94d42978975f7d23338a996125ac0e9"
PATCH_SIZE = 512
BLOCK_SIZE = 2048
GUARD = PATCH_SIZE // 2
SEEDS = base.SEEDS
LABELS = {
    "no damage": 0,
    "minor damage": 1,
    "major damage": 1,
    "destroyed": 1,
}
TRAIN_SITES = [
    {"name": "090403-Lancaster-Canyon-Gate.geo.tif", "gsd_m": 0.036510},
    {"name": "1001-Summerlin-San-Carlos.geo.tif", "gsd_m": 0.040596},
]


def source_paths(name: str) -> tuple[Path, Path]:
    image = hf_hub_download(
        REPO,
        f"train/imagery/UAS/{name}",
        repo_type="dataset",
        revision=SNAPSHOT,
    )
    annotation = hf_hub_download(
        REPO,
        f"train/annotations/UAS/building_damage_assessment/{name}.json",
        repo_type="dataset",
        revision=SNAPSHOT,
    )
    return Path(image), Path(annotation)


def guarded_split(name: str, x: float, y: float) -> str | None:
    residual_x, residual_y = x % BLOCK_SIZE, y % BLOCK_SIZE
    if not (
        GUARD <= residual_x < BLOCK_SIZE - GUARD
        and GUARD <= residual_y < BLOCK_SIZE - GUARD
    ):
        return None
    grid_x, grid_y = int(x // BLOCK_SIZE), int(y // BLOCK_SIZE)
    token = f"{name}:{grid_x}:{grid_y}".encode()
    value = int(hashlib.sha256(token).hexdigest()[:8], 16) % 10
    if value < 6:
        return "train"
    if value < 8:
        return "val"
    return "test"


def prepare_patches(output_root: Path, force: bool = False) -> list[dict]:
    manifest_path = output_root / "patch_manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        audit_overlap(manifest)
        return manifest

    Image.MAX_IMAGE_PIXELS = None
    patch_root = output_root / "patches"
    patch_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for site in TRAIN_SITES:
        name = site["name"]
        image_path, annotation_path = source_paths(name)
        annotations = json.loads(annotation_path.read_text())
        destination = patch_root / Path(name).stem
        destination.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as raster:
            for item in annotations:
                if item.get("label") not in LABELS:
                    continue
                pixels = item.get("pixels") or []
                if not pixels:
                    continue
                x = float(np.mean([point["x"] for point in pixels]))
                y = float(np.mean([point["y"] for point in pixels]))
                split = guarded_split(name, x, y)
                if split is None:
                    continue
                left, top = int(round(x)) - GUARD, int(round(y)) - GUARD
                path = destination / f"{item['building_id']}.jpg"
                if force or not path.exists():
                    raster.crop(
                        (left, top, left + PATCH_SIZE, top + PATCH_SIZE)
                    ).convert("RGB").save(path, quality=92, optimize=True)
                manifest.append(
                    {
                        "path": str(path),
                        "label": LABELS[item["label"]],
                        "source_label": item["label"],
                        "building_id": item["building_id"],
                        "orthomosaic": name,
                        "gsd_m": site["gsd_m"],
                        "centroid": [x, y],
                        "box": [left, top, left + PATCH_SIZE, top + PATCH_SIZE],
                        "split": split,
                        "source_image": str(image_path),
                        "source_annotation": str(annotation_path),
                    }
                )
        print("GUARDED_SITE_COMPLETE", name, len(manifest), flush=True)

    audit = audit_overlap(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (output_root / "split_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    return manifest


def boxes_overlap(a: list[int], b: list[int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def audit_overlap(manifest: list[dict]) -> dict:
    conflicts = []
    for name in sorted({row["orthomosaic"] for row in manifest}):
        rows = [row for row in manifest if row["orthomosaic"] == name]
        bins: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            x, y = row["centroid"]
            bin_x, bin_y = int(x // PATCH_SIZE), int(y // PATCH_SIZE)
            for delta_x in (-1, 0, 1):
                for delta_y in (-1, 0, 1):
                    for other_index in bins[(bin_x + delta_x, bin_y + delta_y)]:
                        other = rows[other_index]
                        if (
                            row["split"] != other["split"]
                            and boxes_overlap(row["box"], other["box"])
                        ):
                            conflicts.append(
                                {
                                    "orthomosaic": name,
                                    "a": row["building_id"],
                                    "b": other["building_id"],
                                    "split_a": row["split"],
                                    "split_b": other["split"],
                                }
                            )
            bins[(bin_x, bin_y)].append(index)
    audit = {
        "patch_size": PATCH_SIZE,
        "block_size": BLOCK_SIZE,
        "guard": GUARD,
        "n_patches": len(manifest),
        "cross_split_intersections": conflicts,
        "w10_crasar_pass": not conflicts,
    }
    if conflicts:
        raise RuntimeError(
            f"{len(conflicts)} cross-split patch intersections remain"
        )
    return audit


def summarize_manifest(manifest: list[dict]) -> dict:
    summary = {}
    for split in ("train", "val", "test"):
        rows = [row for row in manifest if row["split"] == split]
        summary[split] = {
            "n": len(rows),
            "undamaged": sum(row["label"] == 0 for row in rows),
            "damaged": sum(row["label"] == 1 for row in rows),
            "sites": {
                site["name"]: sum(row["orthomosaic"] == site["name"] for row in rows)
                for site in TRAIN_SITES
            },
        }
    return summary


def received_scores(model, rows: list[dict], device) -> dict:
    getter = lambda row: Image.open(row["path"])
    label_getter = lambda row: row["label"]
    probabilities = {}
    labels = None
    for scale in base.SCALES:
        dataset = base.ImageItems(
            rows, getter, label_getter, train=False, scale=scale
        )
        probability, current_labels, _ = base.predict(model, dataset, device)
        probabilities[scale] = probability
        labels = current_labels
    assert labels is not None
    prediction = probabilities[1].argmax(1)
    confidence = probabilities[1].max(1)
    errors = (prediction != labels).astype(int)
    rcg = base.rcg_score(probabilities)
    counts = np.bincount(labels, minlength=2)
    majority = float(counts.max() / counts.sum())
    return {
        "n": len(labels),
        "n_errors": int(errors.sum()),
        "accuracy": float((prediction == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "majority_accuracy": majority,
        "auroc_confidence": base.safe_auroc(errors, 1.0 - confidence),
        "auroc_rcg": base.safe_auroc(errors, rcg),
        "auprc_confidence": float(
            average_precision_score(errors, 1.0 - confidence)
        ),
        "auprc_rcg": float(average_precision_score(errors, rcg)),
    }


def aggregate(seed_results: dict[str, dict]) -> dict:
    metrics = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "majority_accuracy",
        "auroc_confidence",
        "auroc_rcg",
        "auprc_confidence",
        "auprc_rcg",
    )
    output = {}
    for metric in metrics:
        values = np.asarray(
            [seed_results[str(seed)]["internal_test"][metric] for seed in SEEDS],
            dtype=float,
        )
        output[metric] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    output["w11_internal_pass"] = (
        all(
            seed_results[str(seed)]["internal_test"]["accuracy"]
            > seed_results[str(seed)]["internal_test"]["majority_accuracy"]
            for seed in SEEDS
        )
        and output["balanced_accuracy"]["mean"] >= 0.60
    )
    return output


def run(output_root: Path, epochs: int, force_patches: bool) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = prepare_patches(output_root, force=force_patches)
    train_rows = [row for row in manifest if row["split"] == "train"]
    val_rows = [row for row in manifest if row["split"] == "val"]
    test_rows = [row for row in manifest if row["split"] == "test"]
    image_getter = lambda row: Image.open(row["path"])
    label_getter = lambda row: row["label"]
    train_labels = [row["label"] for row in train_rows]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    seed_results = {}
    for seed in SEEDS:
        started = time.time()
        seed_dir = output_root / f"seed_{seed}"
        model, _, best_f1 = base.train_one(
            seed=seed,
            num_labels=2,
            train_dataset=base.ImageItems(
                train_rows, image_getter, label_getter, train=True
            ),
            val_dataset=base.ImageItems(
                val_rows, image_getter, label_getter, train=False
            ),
            train_labels=train_labels,
            device=device,
            epochs=epochs,
            output_dir=seed_dir,
        )
        result = {
            "seed": seed,
            "best_val_macro_f1": best_f1,
            "internal_test": received_scores(model, test_rows, device),
            "runtime_seconds": time.time() - started,
        }
        (seed_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        seed_results[str(seed)] = result
        print("LEAKAGE_FREE_CRASAR_SEED_COMPLETE", seed, json.dumps(result), flush=True)

    summary = {
        "protocol": {
            "dataset": REPO,
            "snapshot": SNAPSHOT,
            "training_sites": TRAIN_SITES,
            "patch_size": PATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "guard": GUARD,
            "seeds": list(SEEDS),
            "epochs": epochs,
            "device": str(device),
        },
        "manifest_summary": summarize_manifest(manifest),
        "split_audit": audit_overlap(manifest),
        "seeds": seed_results,
        "aggregate": aggregate(seed_results),
    }
    output = output_root / "leakage_free_crasar.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print("LEAKAGE_FREE_CRASAR_COMPLETE", output, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--force-patches", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/derived/greatness_iteration3/crasar",
    )
    args = parser.parse_args()
    if args.prepare_only:
        manifest = prepare_patches(args.output, force=args.force_patches)
        print(json.dumps(summarize_manifest(manifest), indent=2))
        return
    run(args.output, args.epochs, args.force_patches)


if __name__ == "__main__":
    main()
