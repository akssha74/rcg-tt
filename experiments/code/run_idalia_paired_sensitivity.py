#!/usr/bin/env python3
"""Third-event paired UAS/crewed measured-GSD sensitivity."""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "imported/experiments/code")
)
import run_primary_multiseed as base


ROOT = Path(__file__).resolve().parents[2]
IMPORTED = ROOT / "experiments/imported/experiments/derived/greatness_iteration3"
OUTPUT = ROOT / "experiments/derived/idalia_paired"
REPO = "CRASAR/CRASAR-U-DROIDs"
REVISION = "47cf4ab3a94d42978975f7d23338a996125ac0e9"
UAS_NAME = "20230830-SteinhatcheeRiver.geo.tif"
CREWED_NAME = f"{UAS_NAME}_20230831a_RGB.geo.tif"
UAS_GSD = 0.127000
CREWED_GSD = 0.150161
PATCH_SIZE = 512
VALID = {"no damage", "minor damage", "major damage", "destroyed"}
LABEL = {"no damage": 0, "minor damage": 1, "major damage": 1, "destroyed": 1}
SEEDS = (101, 202, 303)
BOOTSTRAPS = 2_000


def download(modality, kind, name):
    folder = "imagery" if kind == "image" else "annotations"
    suffix = name if kind == "image" else f"building_damage_assessment/{name}.json"
    return Path(
        hf_hub_download(
            REPO,
            f"test/{folder}/{modality}/{suffix}",
            repo_type="dataset",
            revision=REVISION,
        )
    )


def centroid(item):
    pixels = item.get("pixels") or []
    return (
        float(np.mean([point["x"] for point in pixels])),
        float(np.mean([point["y"] for point in pixels])),
    )


def crop_box(point):
    half = PATCH_SIZE // 2
    left, top = int(round(point[0])) - half, int(round(point[1])) - half
    return left, top, left + PATCH_SIZE, top + PATCH_SIZE


def assign_clusters(rows):
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parent[second] = first

    for first in range(len(rows)):
        for second in range(first):
            uas_a, uas_b = rows[first]["uas_centroid"], rows[second]["uas_centroid"]
            crew_a, crew_b = (
                rows[first]["crewed_centroid"],
                rows[second]["crewed_centroid"],
            )
            uas_overlap = (
                abs(uas_a[0] - uas_b[0]) < PATCH_SIZE
                and abs(uas_a[1] - uas_b[1]) < PATCH_SIZE
            )
            crew_overlap = (
                abs(crew_a[0] - crew_b[0]) < PATCH_SIZE
                and abs(crew_a[1] - crew_b[1]) < PATCH_SIZE
            )
            if uas_overlap or crew_overlap:
                union(first, second)
    identities = {}
    for index in range(len(rows)):
        root = find(index)
        identities.setdefault(root, []).append(index)
    for indexes in identities.values():
        identity = min(rows[index]["building_id"] for index in indexes)
        for index in indexes:
            rows[index]["joint_cluster"] = identity


def prepare():
    manifest_path = OUTPUT / "pair_manifest.json"
    if manifest_path.exists():
        rows = json.loads(manifest_path.read_text())
        if rows and "joint_cluster" in rows[0]:
            return rows
    uas_image = download("UAS", "image", UAS_NAME)
    crew_image = download("CREWED", "image", CREWED_NAME)
    uas_annotation = {
        row["building_id"]: row
        for row in json.loads(download("UAS", "annotation", UAS_NAME).read_text())
    }
    crew_annotation = {
        row["building_id"]: row
        for row in json.loads(
            download("CREWED", "annotation", CREWED_NAME).read_text()
        )
    }
    common = sorted(uas_annotation.keys() & crew_annotation.keys())
    eligible = [
        identity
        for identity in common
        if uas_annotation[identity].get("label") in VALID
        and crew_annotation[identity].get("label")
        == uas_annotation[identity].get("label")
    ]
    selection = {
        "uas_annotations": len(uas_annotation),
        "crewed_annotations": len(crew_annotation),
        "common_building_ids": len(common),
        "same_eligible_label": len(eligible),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    uas_root, crew_root = OUTPUT / "uas", OUTPUT / "crewed"
    uas_root.mkdir(exist_ok=True)
    crew_root.mkdir(exist_ok=True)
    rows = []
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(uas_image) as uas_raster, Image.open(crew_image) as crew_raster:
        for identity in eligible:
            uas_point = centroid(uas_annotation[identity])
            crew_point = centroid(crew_annotation[identity])
            uas_path, crew_path = uas_root / f"{identity}.jpg", crew_root / f"{identity}.jpg"
            if not uas_path.exists():
                uas_raster.crop(crop_box(uas_point)).convert("RGB").save(
                    uas_path, quality=92, optimize=True
                )
            if not crew_path.exists():
                crew_raster.crop(crop_box(crew_point)).convert("RGB").save(
                    crew_path, quality=92, optimize=True
                )
            rows.append(
                {
                    "building_id": identity,
                    "label": LABEL[uas_annotation[identity]["label"]],
                    "source_label": uas_annotation[identity]["label"],
                    "uas_path": str(uas_path),
                    "crewed_path": str(crew_path),
                    "uas_centroid": list(uas_point),
                    "crewed_centroid": list(crew_point),
                }
            )
    assign_clusters(rows)
    manifest_path.write_text(json.dumps(rows, indent=2) + "\n")
    (OUTPUT / "selection_audit.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    return rows


def scores(model, rows, path_key, device):
    getter = lambda row: Image.open(row[path_key])
    label_getter = lambda row: row["label"]
    probabilities = {}
    labels = None
    for scale in (1, 2, 4, 8):
        probabilities[scale], labels, _ = base.predict(
            model,
            base.ImageItems(rows, getter, label_getter, train=False, scale=scale),
            device,
        )
    prediction = probabilities[1].argmax(1)
    confidence = probabilities[1].max(1)
    rcg = base.rcg_score(probabilities)
    return {
        "labels": labels,
        "prediction": prediction,
        "confidence": confidence,
        "rcg": rcg,
    }


def metrics(value):
    labels, prediction = value["labels"], value["prediction"]
    errors = (prediction != labels).astype(int)
    counts = np.bincount(labels, minlength=2)
    return {
        "n": len(labels),
        "n_errors": int(errors.sum()),
        "accuracy": float((prediction == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "majority_accuracy": float(counts.max() / counts.sum()),
        "auroc_confidence": float(roc_auc_score(errors, 1 - value["confidence"])),
        "auroc_rcg": float(roc_auc_score(errors, value["rcg"])),
        "auprc_confidence": float(
            average_precision_score(errors, 1 - value["confidence"])
        ),
        "auprc_rcg": float(average_precision_score(errors, value["rcg"])),
    }


def cluster_bootstrap(rows, uas, crewed, seed):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["joint_cluster"]].append(index)
    identities = sorted(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAPS):
        indexes = []
        for identity in rng.choice(identities, len(identities), replace=True):
            indexes.extend(groups[identity])
        indexes = np.asarray(indexes)
        errors = (
            crewed["prediction"][indexes] != crewed["labels"][indexes]
        ).astype(int)
        if len(np.unique(errors)) < 2:
            continue
        values.append(
            roc_auc_score(errors, crewed["rcg"][indexes])
            - roc_auc_score(errors, 1 - crewed["confidence"][indexes])
        )
    values = np.asarray(values)
    return {
        "clusters": len(identities),
        "mean": float(values.mean()) if len(values) else None,
        "ci95": (
            [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
            if len(values)
            else None
        ),
        "repeats": len(values),
    }


def main():
    started = time.time()
    rows = prepare()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    results = {}
    portable_index = [
        {
            "row": index,
            "building_id": row["building_id"],
            "label": row["label"],
            "source_label": row["source_label"],
            "joint_cluster": row["joint_cluster"],
        }
        for index, row in enumerate(rows)
    ]
    (OUTPUT / "score_index.json").write_text(
        json.dumps(portable_index, indent=2) + "\n"
    )
    for seed in SEEDS:
        checkpoint = IMPORTED / f"crasar/seed_{seed}/best.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = base.build_model(2)
        model.load_state_dict(state["state_dict"])
        model.to(device)
        uas = scores(model, rows, "uas_path", device)
        crewed = scores(model, rows, "crewed_path", device)
        score_path = OUTPUT / f"seed_{seed}.npz"
        np.savez_compressed(
            score_path,
            labels=uas["labels"].astype(np.int64),
            uas_prediction=uas["prediction"].astype(np.int64),
            uas_confidence=uas["confidence"],
            uas_rcg=uas["rcg"],
            crewed_prediction=crewed["prediction"].astype(np.int64),
            crewed_confidence=crewed["confidence"],
            crewed_rcg=crewed["rcg"],
        )
        uas_metrics, crewed_metrics = metrics(uas), metrics(crewed)
        results[str(seed)] = {
            "uas": uas_metrics,
            "crewed": crewed_metrics,
            "paired_accuracy_change": (
                uas_metrics["accuracy"] - crewed_metrics["accuracy"]
            ),
            "crewed_rcg_lift": (
                crewed_metrics["auroc_rcg"]
                - crewed_metrics["auroc_confidence"]
            ),
            "cluster_bootstrap": cluster_bootstrap(
                rows, uas, crewed, 360_000 + seed
            ),
            "score_path": str(score_path.relative_to(ROOT)),
        }
        print("IDALIA_SEED_COMPLETE", seed, json.dumps(results[str(seed)]), flush=True)
    balanced = [results[str(seed)]["uas"]["balanced_accuracy"] for seed in SEEDS]
    summary = {
        "protocol": {
            "dataset": REPO,
            "revision": REVISION,
            "event": "Hurricane Idalia",
            "uas": UAS_NAME,
            "crewed": CREWED_NAME,
            "uas_gsd_m": UAS_GSD,
            "crewed_gsd_m": CREWED_GSD,
            "uas_footprint_m": PATCH_SIZE * UAS_GSD,
            "crewed_footprint_m": PATCH_SIZE * CREWED_GSD,
            "seeds": list(SEEDS),
            "bootstrap_repeats": BOOTSTRAPS,
            "device": str(device),
        },
        "selection": json.loads((OUTPUT / "selection_audit.json").read_text()),
        "results": results,
        "aggregate": {
            "mean_uas_balanced_accuracy": float(np.mean(balanced)),
            "e1_classifier_validity": float(np.mean(balanced)) >= 0.55,
            "e2_completeness": len(results) == 3,
            "mean_crewed_rcg_lift": float(
                np.mean([results[str(seed)]["crewed_rcg_lift"] for seed in SEEDS])
            ),
        },
        "runtime_seconds": time.time() - started,
    }
    output = OUTPUT / "idalia_paired_sensitivity.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print("IDALIA_PAIRED_SENSITIVITY_COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
