#!/usr/bin/env python3
"""Measured-GSD RCG audit on CRASAR-U-DROIDs building patches."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_primary_multiseed as base


ROOT = Path(__file__).resolve().parents[2]
REPO = "CRASAR/CRASAR-U-DROIDs"
SNAPSHOT = "47cf4ab3a94d42978975f7d23338a996125ac0e9"
PATCH_SIZE = 512
BLOCK_SIZE = 2048
SEEDS = base.SEEDS
ORTHOMOSAICS = [
    {
        "name": "090403-Lancaster-Canyon-Gate.geo.tif",
        "split": "train",
        "gsd_m": 0.036510,
    },
    {
        "name": "1001-Harlem-Heights.geo.tif",
        "split": "train",
        "gsd_m": 0.046720,
    },
    {
        "name": "20230830-SteinhatcheeRiver.geo.tif",
        "split": "test",
        "gsd_m": 0.127000,
    },
]
LABELS = {
    "no damage": 0,
    "minor damage": 1,
    "major damage": 1,
    "destroyed": 1,
}


def source_paths(row: dict) -> tuple[Path, Path]:
    name = row["name"]
    split = row["split"]
    image = hf_hub_download(
        REPO,
        f"{split}/imagery/UAS/{name}",
        repo_type="dataset",
        revision=SNAPSHOT,
    )
    annotation = hf_hub_download(
        REPO,
        f"{split}/annotations/UAS/building_damage_assessment/{name}.json",
        repo_type="dataset",
        revision=SNAPSHOT,
    )
    return Path(image), Path(annotation)


def spatial_split(boundary: str, x: float, y: float) -> str:
    grid_x, grid_y = int(x // BLOCK_SIZE), int(y // BLOCK_SIZE)
    token = f"{boundary}:{grid_x}:{grid_y}".encode()
    value = int(hashlib.sha256(token).hexdigest()[:8], 16) % 10
    if value < 6:
        return "train"
    if value < 8:
        return "val"
    return "test"


def prepare_patches(output_root: Path, force: bool = False) -> list[dict]:
    patch_root = output_root / "patches"
    manifest_path = output_root / "patch_manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())
    patch_root.mkdir(parents=True, exist_ok=True)
    Image.MAX_IMAGE_PIXELS = None
    manifest = []
    for row in ORTHOMOSAICS:
        image_path, annotation_path = source_paths(row)
        annotations = json.loads(annotation_path.read_text())
        destination = patch_root / Path(row["name"]).stem
        destination.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as raster:
            width, height = raster.size
            print(
                "ORTHO_OPEN",
                row["name"],
                width,
                height,
                row["gsd_m"],
                flush=True,
            )
            for item in annotations:
                if item.get("label") not in LABELS:
                    continue
                pixels = item.get("pixels") or []
                if not pixels:
                    continue
                centre_x = float(np.mean([point["x"] for point in pixels]))
                centre_y = float(np.mean([point["y"] for point in pixels]))
                half = PATCH_SIZE // 2
                box = (
                    int(round(centre_x)) - half,
                    int(round(centre_y)) - half,
                    int(round(centre_x)) + half,
                    int(round(centre_y)) + half,
                )
                patch_name = f"{item['building_id']}.jpg"
                patch_path = destination / patch_name
                if force or not patch_path.exists():
                    patch = raster.crop(box).convert("RGB")
                    patch.save(patch_path, quality=92, optimize=True)
                manifest.append(
                    {
                        "path": str(patch_path),
                        "label": LABELS[item["label"]],
                        "source_label": item["label"],
                        "building_id": item["building_id"],
                        "orthomosaic": row["name"],
                        "gsd_m": row["gsd_m"],
                        "centroid": [centre_x, centre_y],
                        "split": spatial_split(
                            row["name"], centre_x, centre_y
                        ),
                        "source_image": str(image_path),
                        "source_annotation": str(annotation_path),
                    }
                )
        print(
            "ORTHO_PATCHES_COMPLETE",
            row["name"],
            sum(entry["orthomosaic"] == row["name"] for entry in manifest),
            flush=True,
        )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def summarize_manifest(manifest: list[dict]) -> dict:
    out = {}
    for gsd in sorted({row["gsd_m"] for row in manifest}):
        out[str(gsd)] = {}
        for split in ("train", "val", "test"):
            rows = [
                row
                for row in manifest
                if row["gsd_m"] == gsd and row["split"] == split
            ]
            out[str(gsd)][split] = {
                "n": len(rows),
                "undamaged": sum(row["label"] == 0 for row in rows),
                "damaged": sum(row["label"] == 1 for row in rows),
            }
    return out


def dataset_rows(manifest: list[dict], split: str, gsd: float | None = None):
    return [
        row
        for row in manifest
        if row["split"] == split and (gsd is None or row["gsd_m"] == gsd)
    ]


def evaluate_stratum(model, rows: list[dict], device) -> dict:
    image_getter = lambda item: Image.open(item["path"])
    label_getter = lambda item: item["label"]
    probabilities = {}
    labels = None
    for scale in base.SCALES:
        dataset = base.ImageItems(
            rows, image_getter, label_getter, train=False, scale=scale
        )
        probability, current_labels, _ = base.predict(model, dataset, device)
        probabilities[scale] = probability
        labels = current_labels
    assert labels is not None
    rcg = base.rcg_score(probabilities)
    native_confidence = probabilities[1].max(1)
    native_prediction = probabilities[1].argmax(1)
    errors = (native_prediction != labels).astype(int)
    return {
        "n": len(labels),
        "n_errors": int(errors.sum()),
        "accuracy": float((1 - errors).mean()),
        "error_confidence": (
            float(native_confidence[errors == 1].mean())
            if np.any(errors)
            else None
        ),
        "auroc_confidence": base.safe_auroc(errors, 1.0 - native_confidence),
        "auroc_rcg": base.safe_auroc(errors, rcg),
        "lift_rcg_minus_confidence": (
            base.safe_auroc(errors, rcg)
            - base.safe_auroc(errors, 1.0 - native_confidence)
        ),
    }


def fine_validation_thresholds(model, manifest: list[dict], device):
    finest = min(row["gsd_m"] for row in manifest)
    rows = dataset_rows(manifest, "val", finest)
    image_getter = lambda item: Image.open(item["path"])
    label_getter = lambda item: item["label"]
    probabilities = {}
    for scale in base.SCALES:
        dataset = base.ImageItems(
            rows, image_getter, label_getter, train=False, scale=scale
        )
        probability, _, _ = base.predict(model, dataset, device)
        probabilities[scale] = probability
    return {
        "gsd_m": finest,
        "confidence": float(
            np.quantile(probabilities[1].max(1), 1.0 - base.GAMMA)
        ),
        "rcg": float(np.quantile(base.rcg_score(probabilities), base.GAMMA)),
    }


def transfer_stratum(model, rows: list[dict], thresholds: dict, device) -> dict:
    image_getter = lambda item: Image.open(item["path"])
    label_getter = lambda item: item["label"]
    probabilities = {}
    labels = None
    for scale in base.SCALES:
        dataset = base.ImageItems(
            rows, image_getter, label_getter, train=False, scale=scale
        )
        probability, current_labels, _ = base.predict(model, dataset, device)
        probabilities[scale] = probability
        labels = current_labels
    assert labels is not None
    confidence = probabilities[1].max(1)
    prediction = probabilities[1].argmax(1)
    rcg = base.rcg_score(probabilities)
    false_critical = (labels == 0) & (prediction == 1)

    def summary(mask):
        return {
            "coverage": float(mask.mean()),
            "fcr": float(false_critical[mask].mean()) if np.any(mask) else None,
            "n": int(mask.sum()),
        }

    return {
        "confidence": summary(confidence >= thresholds["confidence"]),
        "rcg": summary(rcg <= thresholds["rcg"]),
    }


def aggregate(seed_results: dict[str, dict]) -> dict:
    gsds = sorted(
        {
            float(key)
            for result in seed_results.values()
            for key in result["strata"]
        }
    )
    out = {"strata": {}}
    for gsd in gsds:
        key = str(gsd)
        out["strata"][key] = {}
        for metric in (
            "accuracy",
            "error_confidence",
            "auroc_confidence",
            "auroc_rcg",
            "lift_rcg_minus_confidence",
        ):
            values = np.asarray(
                [seed_results[str(seed)]["strata"][key][metric] for seed in SEEDS],
                dtype=float,
            )
            out["strata"][key][metric] = {
                "values": values.tolist(),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }
    finest, coarsest = str(min(gsds)), str(max(gsds))
    out["w6_accuracy_drop"] = (
        out["strata"][finest]["accuracy"]["mean"]
        - out["strata"][coarsest]["accuracy"]["mean"]
    )
    out["w6_coarse_rcg_lift"] = out["strata"][coarsest][
        "lift_rcg_minus_confidence"
    ]["mean"]
    out["w6_pass"] = (
        out["w6_accuracy_drop"] > 0 and out["w6_coarse_rcg_lift"] > 0
    )
    return out


def run(output_root: Path, epochs: int, force_patches: bool = False) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = prepare_patches(output_root, force=force_patches)
    manifest_summary = summarize_manifest(manifest)
    print("PATCH_MANIFEST", json.dumps(manifest_summary, sort_keys=True), flush=True)
    image_getter = lambda item: Image.open(item["path"])
    label_getter = lambda item: item["label"]
    train_rows = dataset_rows(manifest, "train")
    val_rows = dataset_rows(manifest, "val")
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
        thresholds = fine_validation_thresholds(model, manifest, device)
        strata = {}
        transfer = {}
        for gsd in sorted({row["gsd_m"] for row in manifest}):
            rows = dataset_rows(manifest, "test", gsd)
            strata[str(gsd)] = evaluate_stratum(model, rows, device)
            transfer[str(gsd)] = transfer_stratum(
                model, rows, thresholds, device
            )
        result = {
            "seed": seed,
            "best_val_macro_f1": best_f1,
            "fine_validation_thresholds": thresholds,
            "strata": strata,
            "transfer": transfer,
            "runtime_seconds": time.time() - started,
        }
        (seed_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        seed_results[str(seed)] = result
        print("CRASAR_SEED_COMPLETE", seed, json.dumps(strata), flush=True)
    summary = {
        "protocol": {
            "dataset": REPO,
            "snapshot": SNAPSHOT,
            "seeds": list(SEEDS),
            "patch_size": PATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "epochs": epochs,
            "orthomosaics": ORTHOMOSAICS,
        },
        "manifest_summary": manifest_summary,
        "seeds": seed_results,
        "aggregate": aggregate(seed_results),
    }
    output = output_root / "measured_gsd_crasar.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print("MEASURED_GSD_COMPLETE", output, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--force-patches", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/derived/greatness_strengthening/crasar",
    )
    args = parser.parse_args()
    if args.prepare_only:
        manifest = prepare_patches(args.output, force=args.force_patches)
        print(json.dumps(summarize_manifest(manifest), indent=2))
        return
    run(args.output, args.epochs, force_patches=args.force_patches)


if __name__ == "__main__":
    main()
