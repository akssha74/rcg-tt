#!/usr/bin/env python3
"""Held-out multi-event paired measured-GSD evaluation."""

from __future__ import annotations

import argparse
import json
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_leakage_free_crasar as training
import run_primary_multiseed as base


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments/derived/greatness_iteration3/crasar"
GAMMAS = (0.5, 0.7, 0.9)
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 260724
VALID = set(training.LABELS)
SITES = [
    {
        "id": "harlem-heights",
        "source_split": "train",
        "event": "Hurricane Ian",
        "uas_name": "1001-Harlem-Heights.geo.tif",
        "sat_name": (
            "1001-Harlem-Heights.geo.tif_"
            "10300100DB06A700-visual.tif.geo.tif"
        ),
        "uas_gsd_m": 0.046720,
        "satellite_gsd_m": 0.305175781,
    },
    {
        "id": "mcgregor-college-parkway-south-1",
        "source_split": "train",
        "event": "Hurricane Ian",
        "uas_name": "1001-McGregor-College-Pkwy-South.1.geo.tif",
        "sat_name": (
            "1001-McGregor-College-Pkwy-South.1.geo.tif_"
            "10300100DB06A700-visual.tif.geo.tif"
        ),
        "uas_gsd_m": 0.0383888,
        "satellite_gsd_m": 0.305175781,
    },
    {
        "id": "mexico-beach-2018-10-13",
        "source_split": "test",
        "event": "Hurricane Michael",
        "uas_name": "10132018-MexicoBeach.geo.tif",
        "sat_name": (
            "10132018-MexicoBeach.geo.tif_"
            "104001004384D900.tif.geo.tif"
        ),
        "uas_gsd_m": 0.019646,
        "satellite_gsd_m": 0.589669,
    },
    {
        "id": "mexico-beach-2018-10-14",
        "source_split": "test",
        "event": "Hurricane Michael",
        "uas_name": "10142018-MexicoBeach.geo.tif",
        "sat_name": (
            "10142018-MexicoBeach.geo.tif_"
            "104001004384D900.tif.geo.tif"
        ),
        "uas_gsd_m": 0.019465,
        "satellite_gsd_m": 0.589669,
    },
]


def download(site: dict, modality: str, kind: str) -> Path:
    name = site["uas_name"] if modality == "UAS" else site["sat_name"]
    folder = "imagery" if kind == "image" else "annotations"
    suffix = name if kind == "image" else f"building_damage_assessment/{name}.json"
    return Path(
        hf_hub_download(
            training.REPO,
            f"{site['source_split']}/{folder}/{modality}/{suffix}",
            repo_type="dataset",
            revision=training.SNAPSHOT,
        )
    )


def crop_box(item: dict) -> tuple[list[int], tuple[float, float]]:
    pixels = item.get("pixels") or []
    x = float(np.mean([point["x"] for point in pixels]))
    y = float(np.mean([point["y"] for point in pixels]))
    left, top = int(round(x)) - training.GUARD, int(round(y)) - training.GUARD
    return (
        [left, top, left + training.PATCH_SIZE, top + training.PATCH_SIZE],
        (x, y),
    )


def prepare_site(site: dict, force: bool = False) -> list[dict]:
    site_root = OUTPUT / "paired_evaluation" / site["id"]
    manifest_path = site_root / "pair_manifest.json"
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text())
        if existing and all(
            "satellite_centroid" in row and "uas_centroid" in row
            for row in existing
        ):
            return existing

    uas_image = download(site, "UAS", "image")
    sat_image = download(site, "SATELLITE", "image")
    uas_annotation = download(site, "UAS", "annotation")
    sat_annotation = download(site, "SATELLITE", "annotation")
    uas = {row["building_id"]: row for row in json.loads(uas_annotation.read_text())}
    sat = {row["building_id"]: row for row in json.loads(sat_annotation.read_text())}
    common_ids = sorted(uas.keys() & sat.keys())
    selection_audit = {
        "uas_annotations": len(uas),
        "satellite_annotations": len(sat),
        "common_building_ids": len(common_ids),
        "uas_eligible_labels": sum(
            uas[building_id].get("label") in VALID for building_id in common_ids
        ),
        "satellite_eligible_labels": sum(
            sat[building_id].get("label") in VALID for building_id in common_ids
        ),
        "same_eligible_label": sum(
            uas[building_id].get("label") in VALID
            and sat[building_id].get("label") == uas[building_id].get("label")
            for building_id in common_ids
        ),
    }
    uas_dir, sat_dir = site_root / "uas", site_root / "satellite"
    uas_dir.mkdir(parents=True, exist_ok=True)
    sat_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(uas_image) as uas_raster, Image.open(sat_image) as sat_raster:
        for building_id in common_ids:
            uas_item, sat_item = uas[building_id], sat[building_id]
            if (
                uas_item.get("label") not in VALID
                or sat_item.get("label") != uas_item.get("label")
                or not uas_item.get("pixels")
                or not sat_item.get("pixels")
            ):
                continue
            uas_box, (uas_x, uas_y) = crop_box(uas_item)
            sat_box, (sat_x, sat_y) = crop_box(sat_item)
            uas_path = uas_dir / f"{building_id}.jpg"
            sat_path = sat_dir / f"{building_id}.jpg"
            if force or not uas_path.exists():
                uas_raster.crop(uas_box).convert("RGB").save(
                    uas_path, quality=92, optimize=True
                )
            if force or not sat_path.exists():
                sat_raster.crop(sat_box).convert("RGB").save(
                    sat_path, quality=92, optimize=True
                )
            rows.append(
                {
                    "site": site["id"],
                    "event": site["event"],
                    "building_id": building_id,
                    "spatial_block": (
                        f"{site['id']}:{int(uas_x // training.BLOCK_SIZE)}:"
                        f"{int(uas_y // training.BLOCK_SIZE)}"
                    ),
                    "satellite_spatial_block": (
                        f"{site['id']}:{int(sat_x // training.BLOCK_SIZE)}:"
                        f"{int(sat_y // training.BLOCK_SIZE)}"
                    ),
                    "uas_centroid": [uas_x, uas_y],
                    "satellite_centroid": [sat_x, sat_y],
                    "label": training.LABELS[uas_item["label"]],
                    "source_label": uas_item["label"],
                    "uas_path": str(uas_path),
                    "satellite_path": str(sat_path),
                    "uas_gsd_m": site["uas_gsd_m"],
                    "satellite_gsd_m": site["satellite_gsd_m"],
                }
            )
    manifest_path.write_text(json.dumps(rows, indent=2) + "\n")
    (site_root / "selection_audit.json").write_text(
        json.dumps(selection_audit, indent=2) + "\n"
    )
    print("PAIRED_SITE_PREPARED", site["id"], len(rows), flush=True)
    return rows


def assign_joint_spatial_clusters(rows: list[dict]) -> None:
    """Group pairs when either UAS or satellite 512-pixel crops overlap."""
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first in range(len(rows)):
        for second in range(first):
            if rows[first]["site"] != rows[second]["site"]:
                continue
            uas_first, uas_second = (
                rows[first]["uas_centroid"],
                rows[second]["uas_centroid"],
            )
            sat_first, sat_second = (
                rows[first]["satellite_centroid"],
                rows[second]["satellite_centroid"],
            )
            uas_overlap = (
                abs(uas_first[0] - uas_second[0]) < training.PATCH_SIZE
                and abs(uas_first[1] - uas_second[1]) < training.PATCH_SIZE
            )
            satellite_overlap = (
                abs(sat_first[0] - sat_second[0]) < training.PATCH_SIZE
                and abs(sat_first[1] - sat_second[1]) < training.PATCH_SIZE
            )
            if uas_overlap or satellite_overlap:
                union(first, second)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    for indexes in groups.values():
        identity = min(rows[index]["building_id"] for index in indexes)
        site = rows[indexes[0]]["site"]
        for index in indexes:
            rows[index]["joint_spatial_cluster"] = f"{site}:{identity}"


def score_rows(model, rows: list[dict], path_key: str, device) -> dict:
    getter = lambda row: Image.open(row[path_key])
    label_getter = lambda row: row["label"]
    probabilities = {}
    features = None
    labels = None
    for scale in base.SCALES:
        dataset = base.ImageItems(
            rows, getter, label_getter, train=False, scale=scale
        )
        probability, current_labels, current_features = base.predict(
            model, dataset, device
        )
        probabilities[scale] = probability
        if scale == 1:
            features = current_features
        labels = current_labels
    assert labels is not None and features is not None
    prediction = probabilities[1].argmax(1)
    confidence = probabilities[1].max(1)
    return {
        "labels": labels,
        "prediction": prediction,
        "confidence": confidence,
        "rcg": base.rcg_score(probabilities),
        "features": features,
    }


def fit_thresholds(scores: dict, gammas: tuple[float, ...]) -> dict:
    return {
        str(gamma): {
            "confidence": float(np.quantile(scores["confidence"], 1.0 - gamma)),
            "rcg": float(np.quantile(scores["rcg"], gamma)),
        }
        for gamma in gammas
    }


def metric_summary(scores: dict) -> dict:
    labels = scores["labels"]
    prediction = scores["prediction"]
    errors = (prediction != labels).astype(int)
    counts = np.bincount(labels, minlength=2)
    return {
        "n": len(labels),
        "n_errors": int(errors.sum()),
        "accuracy": float((prediction == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "majority_accuracy": float(counts.max() / counts.sum()),
        "auroc_confidence": base.safe_auroc(errors, 1.0 - scores["confidence"]),
        "auroc_rcg": base.safe_auroc(errors, scores["rcg"]),
        "auprc_confidence": float(
            average_precision_score(errors, 1.0 - scores["confidence"])
        ),
        "auprc_rcg": float(average_precision_score(errors, scores["rcg"])),
    }


def transfer_summary(scores: dict, thresholds: dict) -> dict:
    labels, prediction = scores["labels"], scores["prediction"]
    errors = prediction != labels
    false_critical = (labels == 0) & (prediction == 1)
    output = {}
    for gamma in GAMMAS:
        current = thresholds[str(gamma)]
        masks = {
            "confidence": scores["confidence"] >= current["confidence"],
            "rcg": scores["rcg"] <= current["rcg"],
        }
        output[str(gamma)] = {}
        for gate, mask in masks.items():
            output[str(gamma)][gate] = {
                "coverage": float(mask.mean()),
                "absolute_coverage_error": float(abs(mask.mean() - gamma)),
                "selective_risk": float(errors[mask].mean()) if np.any(mask) else None,
                "false_critical_rate": (
                    float(false_critical[mask].mean()) if np.any(mask) else None
                ),
                "n": int(mask.sum()),
            }
    return output


def subset_scores(scores: dict, indexes: np.ndarray) -> dict:
    return {
        key: value[indexes]
        for key, value in scores.items()
        if key in {"labels", "prediction", "confidence", "rcg"}
    }


def spatial_cluster_bootstrap(
    rows: list[dict],
    uas: dict,
    satellite: dict,
    thresholds: dict,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    sites = sorted({row["site"] for row in rows})
    indexes_by_site_block: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, row in enumerate(rows):
        indexes_by_site_block[row["site"]][row["joint_spatial_cluster"]].append(
            index
        )
    values = {
        "accuracy_drop": [],
        "satellite_auroc_lift": [],
        "coverage_difference_gamma_0.7": [],
        "selective_risk_difference_gamma_0.7": [],
    }
    for _ in range(BOOTSTRAP_REPEATS):
        sampled = []
        for site in rng.choice(sites, size=len(sites), replace=True):
            blocks = sorted(indexes_by_site_block[site])
            for block in rng.choice(blocks, size=len(blocks), replace=True):
                sampled.extend(indexes_by_site_block[site][block])
        indexes = np.asarray(sampled, dtype=int)
        uas_sample = subset_scores(uas, indexes)
        sat_sample = subset_scores(satellite, indexes)
        uas_correct = uas_sample["prediction"] == uas_sample["labels"]
        sat_correct = sat_sample["prediction"] == sat_sample["labels"]
        values["accuracy_drop"].append(
            float(uas_correct.mean() - sat_correct.mean())
        )
        sat_errors = (sat_sample["prediction"] != sat_sample["labels"]).astype(int)
        if len(np.unique(sat_errors)) < 2:
            continue
        values["satellite_auroc_lift"].append(
            float(
                roc_auc_score(sat_errors, sat_sample["rcg"])
                - roc_auc_score(sat_errors, 1.0 - sat_sample["confidence"])
            )
        )
        current = thresholds["0.7"]
        confidence_mask = sat_sample["confidence"] >= current["confidence"]
        rcg_mask = sat_sample["rcg"] <= current["rcg"]
        values["coverage_difference_gamma_0.7"].append(
            float(rcg_mask.mean() - confidence_mask.mean())
        )
        if np.any(confidence_mask) and np.any(rcg_mask):
            values["selective_risk_difference_gamma_0.7"].append(
                float(
                    sat_errors[rcg_mask].mean()
                    - sat_errors[confidence_mask].mean()
                )
            )
    return {
        metric: {
            "mean": float(np.mean(samples)),
            "ci95": [
                float(np.percentile(samples, 2.5)),
                float(np.percentile(samples, 97.5)),
            ],
            "repeats": len(samples),
        }
        for metric, samples in values.items()
        if samples
    }


def concatenate(parts: list[dict]) -> dict:
    return {
        key: np.concatenate([part[key] for part in parts])
        for key in ("labels", "prediction", "confidence", "rcg")
    }


def run(force_patches: bool = False) -> dict:
    site_rows = {site["id"]: prepare_site(site, force_patches) for site in SITES}
    rows = [row for site in SITES for row in site_rows[site["id"]]]
    assign_joint_spatial_clusters(rows)
    score_root = OUTPUT / "paired_scores"
    score_root.mkdir(parents=True, exist_ok=True)
    portable_index = [
        {
            "row": index,
            "site": row["site"],
            "event": row["event"],
            "building_id": row["building_id"],
            "spatial_block": row["spatial_block"],
            "satellite_spatial_block": row["satellite_spatial_block"],
            "joint_spatial_cluster": row["joint_spatial_cluster"],
            "label": row["label"],
            "uas_gsd_m": row["uas_gsd_m"],
            "satellite_gsd_m": row["satellite_gsd_m"],
        }
        for index, row in enumerate(rows)
    ]
    (score_root / "paired_score_index.json").write_text(
        json.dumps(portable_index, indent=2) + "\n"
    )
    training_manifest = json.loads((OUTPUT / "patch_manifest.json").read_text())
    val_rows = [row for row in training_manifest if row["split"] == "val"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    seed_results = {}
    for seed in base.SEEDS:
        state = torch.load(
            OUTPUT / f"seed_{seed}/best.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = base.build_model(2)
        model.load_state_dict(state["state_dict"])
        model.to(device)
        validation = score_rows(model, val_rows, "path", device)
        thresholds = fit_thresholds(validation, GAMMAS)
        site_scores = {}
        uas_parts, satellite_parts = [], []
        for site_index, site in enumerate(SITES):
            current_rows = site_rows[site["id"]]
            uas = score_rows(model, current_rows, "uas_path", device)
            satellite = score_rows(
                model, current_rows, "satellite_path", device
            )
            uas_parts.append(uas)
            satellite_parts.append(satellite)
            site_scores[site["id"]] = {
                "uas": metric_summary(uas),
                "satellite": metric_summary(satellite),
                "uas_transfer": transfer_summary(uas, thresholds),
                "satellite_transfer": transfer_summary(satellite, thresholds),
                "spatial_cluster_bootstrap": spatial_cluster_bootstrap(
                    current_rows,
                    uas,
                    satellite,
                    thresholds,
                    BOOTSTRAP_SEED + seed + 10_000 * (site_index + 1),
                ),
            }
        pooled_uas, pooled_satellite = concatenate(uas_parts), concatenate(
            satellite_parts
        )
        score_path = score_root / f"seed_{seed}.npz"
        np.savez_compressed(
            score_path,
            labels=pooled_uas["labels"].astype(np.int64),
            uas_prediction=pooled_uas["prediction"].astype(np.int64),
            uas_confidence=pooled_uas["confidence"],
            uas_rcg=pooled_uas["rcg"],
            satellite_prediction=pooled_satellite["prediction"].astype(np.int64),
            satellite_confidence=pooled_satellite["confidence"],
            satellite_rcg=pooled_satellite["rcg"],
        )
        seed_results[str(seed)] = {
            "seed": seed,
            "thresholds": thresholds,
            "sites": site_scores,
            "pooled": {
                "uas": metric_summary(pooled_uas),
                "satellite": metric_summary(pooled_satellite),
                "uas_transfer": transfer_summary(pooled_uas, thresholds),
                "satellite_transfer": transfer_summary(
                    pooled_satellite, thresholds
                ),
            },
            "spatial_cluster_bootstrap": spatial_cluster_bootstrap(
                rows,
                pooled_uas,
                pooled_satellite,
                thresholds,
                BOOTSTRAP_SEED + seed,
            ),
            "per_example_scores": str(score_path.relative_to(ROOT)),
        }
        print(
            "LEAKAGE_FREE_PAIRED_SEED_COMPLETE",
            seed,
            json.dumps(seed_results[str(seed)]["pooled"]),
            flush=True,
        )

    pooled_uas_accuracy = [
        seed_results[str(seed)]["pooled"]["uas"]["accuracy"] for seed in base.SEEDS
    ]
    pooled_uas_majority = [
        seed_results[str(seed)]["pooled"]["uas"]["majority_accuracy"]
        for seed in base.SEEDS
    ]
    balanced = [
        seed_results[str(seed)]["pooled"]["uas"]["balanced_accuracy"]
        for seed in base.SEEDS
    ]
    satellite_lifts = [
        seed_results[str(seed)]["pooled"]["satellite"]["auroc_rcg"]
        - seed_results[str(seed)]["pooled"]["satellite"]["auroc_confidence"]
        for seed in base.SEEDS
    ]
    site_lifts = {
        site["id"]: float(
            np.mean(
                [
                    seed_results[str(seed)]["sites"][site["id"]]["satellite"][
                        "auroc_rcg"
                    ]
                    - seed_results[str(seed)]["sites"][site["id"]]["satellite"][
                        "auroc_confidence"
                    ]
                    for seed in base.SEEDS
                ]
            )
        )
        for site in SITES
    }
    bootstrap_lower = [
        seed_results[str(seed)]["spatial_cluster_bootstrap"][
            "satellite_auroc_lift"
        ]["ci95"][0]
        for seed in base.SEEDS
    ]
    transfer_checks = []
    for seed in base.SEEDS:
        transfer = seed_results[str(seed)]["pooled"]["satellite_transfer"]["0.7"]
        transfer_checks.append(
            transfer["rcg"]["absolute_coverage_error"]
            <= transfer["confidence"]["absolute_coverage_error"]
            and transfer["rcg"]["selective_risk"]
            <= transfer["confidence"]["selective_risk"]
        )
    criteria = {
        "w11_classifier_validity": (
            all(a > b for a, b in zip(pooled_uas_accuracy, pooled_uas_majority))
            and float(np.mean(balanced)) >= 0.60
        ),
        "w12_deployable_ranking": (
            float(np.mean(satellite_lifts)) > 0
            and all(value > 0 for value in bootstrap_lower)
            and sum(value > 0 for value in site_lifts.values()) >= 3
        ),
        "w13_threshold_transfer": all(transfer_checks),
    }
    summary = {
        "protocol": {
            "dataset": training.REPO,
            "snapshot": training.SNAPSHOT,
            "sites": SITES,
            "seeds": list(base.SEEDS),
            "relative_scales": list(base.SCALES),
            "gammas": list(GAMMAS),
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "analysis_status": (
                "post-hoc reproducibility strengthening: adds joint UAS/satellite "
                "overlap clusters, site intervals, selection counts, and raw scores"
            ),
            "cluster_definition": (
                "connected components of pairs whose 512-pixel UAS or satellite "
                "crop rectangles overlap within site"
            ),
            "device": str(device),
        },
        "pair_counts": {
            site["id"]: {
                "n": len(site_rows[site["id"]]),
                "undamaged": sum(
                    row["label"] == 0 for row in site_rows[site["id"]]
                ),
                "damaged": sum(
                    row["label"] == 1 for row in site_rows[site["id"]]
                ),
                "spatial_blocks": len(
                    {row["spatial_block"] for row in site_rows[site["id"]]}
                ),
                "satellite_spatial_blocks": len(
                    {
                        row["satellite_spatial_block"]
                        for row in site_rows[site["id"]]
                    }
                ),
                "joint_spatial_clusters": len(
                    {
                        row["joint_spatial_cluster"]
                        for row in site_rows[site["id"]]
                    }
                ),
            }
            for site in SITES
        },
        "selection_audits": {
            site["id"]: json.loads(
                (
                    OUTPUT
                    / "paired_evaluation"
                    / site["id"]
                    / "selection_audit.json"
                ).read_text()
            )
            for site in SITES
        },
        "seeds": seed_results,
        "aggregate": {
            "pooled_uas_accuracy": pooled_uas_accuracy,
            "pooled_uas_majority_accuracy": pooled_uas_majority,
            "pooled_uas_balanced_accuracy": balanced,
            "satellite_auroc_lifts": satellite_lifts,
            "mean_satellite_auroc_lift": float(np.mean(satellite_lifts)),
            "site_mean_satellite_auroc_lifts": site_lifts,
            "criteria": criteria,
        },
    }
    output = OUTPUT / "leakage_free_paired_gsd_v2.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print("LEAKAGE_FREE_PAIRED_COMPLETE", output, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-patches", action="store_true")
    args = parser.parse_args()
    run(args.force_patches)


if __name__ == "__main__":
    main()
