#!/usr/bin/env python3
"""Paired same-building UAS/satellite measured-GSD RCG evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_primary_multiseed as base


ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "experiments/derived/greatness_strengthening"
CRASAR = DERIVED / "crasar"
SNAPSHOT = "47cf4ab3a94d42978975f7d23338a996125ac0e9"
UAS_NAME = "1001-Harlem-Heights.geo.tif"
SAT_NAME = (
    "1001-Harlem-Heights.geo.tif_"
    "10300100DB06A700-visual.tif.geo.tif"
)
UAS_GSD = 0.046720
SAT_GSD = 0.305175781
VALID = {"no damage", "minor damage", "major damage", "destroyed"}


def prepare_pairs(force: bool = False) -> list[dict]:
    output = CRASAR / "paired_gsd"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "pair_manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())

    uas_manifest = json.loads((CRASAR / "patch_manifest.json").read_text())
    uas_rows = {
        row["building_id"]: row
        for row in uas_manifest
        if row["orthomosaic"] == UAS_NAME and row["split"] in {"val", "test"}
    }
    sat_image = Path(
        hf_hub_download(
            "CRASAR/CRASAR-U-DROIDs",
            f"train/imagery/SATELLITE/{SAT_NAME}",
            repo_type="dataset",
            revision=SNAPSHOT,
        )
    )
    sat_annotation = Path(
        hf_hub_download(
            "CRASAR/CRASAR-U-DROIDs",
            f"train/annotations/SATELLITE/building_damage_assessment/{SAT_NAME}.json",
            repo_type="dataset",
            revision=SNAPSHOT,
        )
    )
    uas_annotation = Path(
        hf_hub_download(
            "CRASAR/CRASAR-U-DROIDs",
            f"train/annotations/UAS/building_damage_assessment/{UAS_NAME}.json",
            repo_type="dataset",
            revision=SNAPSHOT,
        )
    )
    sat = {row["building_id"]: row for row in json.loads(sat_annotation.read_text())}
    uas = {row["building_id"]: row for row in json.loads(uas_annotation.read_text())}
    patch_dir = output / "satellite_patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(sat_image) as raster:
        for building_id, uas_row in uas_rows.items():
            if building_id not in sat or building_id not in uas:
                continue
            sat_item, uas_item = sat[building_id], uas[building_id]
            if (
                sat_item.get("label") not in VALID
                or sat_item.get("label") != uas_item.get("label")
            ):
                continue
            pixels = sat_item.get("pixels") or []
            if not pixels:
                continue
            centre_x = float(np.mean([point["x"] for point in pixels]))
            centre_y = float(np.mean([point["y"] for point in pixels]))
            half = 256
            patch_path = patch_dir / f"{building_id}.jpg"
            if force or not patch_path.exists():
                patch = raster.crop(
                    (
                        int(round(centre_x)) - half,
                        int(round(centre_y)) - half,
                        int(round(centre_x)) + half,
                        int(round(centre_y)) + half,
                    )
                ).convert("RGB")
                patch.save(patch_path, quality=92, optimize=True)
            pairs.append(
                {
                    "building_id": building_id,
                    "label": uas_row["label"],
                    "source_label": uas_item["label"],
                    "split": uas_row["split"],
                    "uas_path": uas_row["path"],
                    "satellite_path": str(patch_path),
                    "uas_gsd_m": UAS_GSD,
                    "satellite_gsd_m": SAT_GSD,
                }
            )
    manifest_path.write_text(json.dumps(pairs, indent=2) + "\n")
    return pairs


def modality_scores(model, rows: list[dict], path_key: str, device) -> dict:
    getter = lambda row: Image.open(row[path_key])
    label = lambda row: row["label"]
    probabilities = {}
    labels = None
    for scale in base.SCALES:
        dataset = base.ImageItems(rows, getter, label, train=False, scale=scale)
        probability, current_labels, _ = base.predict(model, dataset, device)
        probabilities[scale] = probability
        labels = current_labels
    assert labels is not None
    confidence = probabilities[1].max(1)
    prediction = probabilities[1].argmax(1)
    errors = (prediction != labels).astype(int)
    rcg = base.rcg_score(probabilities)
    return {
        "probabilities": probabilities,
        "labels": labels,
        "prediction": prediction,
        "confidence": confidence,
        "rcg": rcg,
        "metrics": {
            "n": len(labels),
            "n_errors": int(errors.sum()),
            "accuracy": float((1 - errors).mean()),
            "error_confidence": (
                float(confidence[errors == 1].mean()) if np.any(errors) else None
            ),
            "auroc_confidence": base.safe_auroc(errors, 1.0 - confidence),
            "auroc_rcg": base.safe_auroc(errors, rcg),
            "lift_rcg_minus_confidence": (
                base.safe_auroc(errors, rcg)
                - base.safe_auroc(errors, 1.0 - confidence)
            ),
        },
    }


def transfer(calibration: dict, evaluation: dict) -> dict:
    conf_threshold = float(
        np.quantile(calibration["confidence"], 1.0 - base.GAMMA)
    )
    rcg_threshold = float(np.quantile(calibration["rcg"], base.GAMMA))
    false_critical = (evaluation["labels"] == 0) & (
        evaluation["prediction"] == 1
    )

    def summarize(mask):
        return {
            "coverage": float(mask.mean()),
            "fcr": float(false_critical[mask].mean()) if np.any(mask) else None,
            "n": int(mask.sum()),
        }

    return {
        "confidence_threshold": conf_threshold,
        "rcg_threshold": rcg_threshold,
        "confidence": summarize(evaluation["confidence"] >= conf_threshold),
        "rcg": summarize(evaluation["rcg"] <= rcg_threshold),
    }


def aggregate(seed_results: dict[str, dict]) -> dict:
    out = {}
    for modality in ("uas", "satellite"):
        out[modality] = {}
        for metric in (
            "accuracy",
            "error_confidence",
            "auroc_confidence",
            "auroc_rcg",
            "lift_rcg_minus_confidence",
        ):
            values = np.asarray(
                [
                    seed_results[str(seed)][modality][metric]
                    for seed in base.SEEDS
                ],
                dtype=float,
            )
            out[modality][metric] = {
                "values": values.tolist(),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }
    out["paired_accuracy_drop"] = (
        out["uas"]["accuracy"]["mean"] - out["satellite"]["accuracy"]["mean"]
    )
    out["satellite_rcg_lift"] = out["satellite"][
        "lift_rcg_minus_confidence"
    ]["mean"]
    out["w6b_pass"] = (
        out["paired_accuracy_drop"] > 0 and out["satellite_rcg_lift"] > 0
    )
    return out


def main() -> None:
    pairs = prepare_pairs()
    counts = {
        split: {
            "n": sum(row["split"] == split for row in pairs),
            "undamaged": sum(
                row["split"] == split and row["label"] == 0 for row in pairs
            ),
            "damaged": sum(
                row["split"] == split and row["label"] == 1 for row in pairs
            ),
        }
        for split in ("val", "test")
    }
    print("PAIRED_COUNTS", json.dumps(counts), flush=True)
    val_rows = [row for row in pairs if row["split"] == "val"]
    test_rows = [row for row in pairs if row["split"] == "test"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    seed_results = {}
    for seed in base.SEEDS:
        checkpoint = CRASAR / f"seed_{seed}/best.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = base.build_model(2)
        model.load_state_dict(state["state_dict"])
        model.to(device)
        calibration = modality_scores(model, val_rows, "uas_path", device)
        uas = modality_scores(model, test_rows, "uas_path", device)
        satellite = modality_scores(
            model, test_rows, "satellite_path", device
        )
        result = {
            "seed": seed,
            "uas": uas["metrics"],
            "satellite": satellite["metrics"],
            "transfer_uas": transfer(calibration, uas),
            "transfer_satellite": transfer(calibration, satellite),
        }
        seed_results[str(seed)] = result
        print("PAIRED_SEED_COMPLETE", seed, json.dumps(result), flush=True)
    summary = {
        "protocol": {
            "dataset": "CRASAR/CRASAR-U-DROIDs",
            "snapshot": SNAPSHOT,
            "area": "1001-Harlem-Heights",
            "date": "2022-09-30",
            "uas_gsd_m": UAS_GSD,
            "satellite_gsd_m": SAT_GSD,
            "same_label_required": True,
            "seeds": list(base.SEEDS),
        },
        "pair_counts": counts,
        "seeds": seed_results,
        "aggregate": aggregate(seed_results),
    }
    output = CRASAR / "paired_measured_gsd.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print("PAIRED_GSD_COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
