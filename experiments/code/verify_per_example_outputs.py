#!/usr/bin/env python3
"""Recompute headline metrics and omitted comparisons from per-example arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/imported/experiments/derived/greatness_iteration3"
OUTPUT = ROOT / "experiments/derived/per_example_verification.json"
REPEATS = 10_000


def weighted_auc_batch(labels, score, counts):
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    sorted_labels = labels[order]
    sorted_counts = counts[:, order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_score) != 0) + 1]
    positive = sorted_counts * sorted_labels
    negative = sorted_counts * (1 - sorted_labels)
    positive_group = np.add.reduceat(positive, starts, axis=1)
    negative_group = np.add.reduceat(negative, starts, axis=1)
    negative_before = np.cumsum(negative_group, axis=1) - negative_group
    concordant = (
        positive_group * (negative_before + 0.5 * negative_group)
    ).sum(axis=1)
    positive_total = positive.sum(axis=1)
    negative_total = negative.sum(axis=1)
    valid = (positive_total > 0) & (negative_total > 0)
    auc = np.full(len(counts), np.nan)
    auc[valid] = concordant[valid] / (
        positive_total[valid] * negative_total[valid]
    )
    return auc, valid


def paired_bootstrap(labels, first, second, seed):
    rng = np.random.default_rng(seed)
    probability = np.full(len(labels), 1.0 / len(labels))
    values = []
    while len(values) < REPEATS:
        size = min(250, REPEATS - len(values))
        counts = rng.multinomial(len(labels), probability, size=size)
        first_auc, first_valid = weighted_auc_batch(labels, first, counts)
        second_auc, second_valid = weighted_auc_batch(labels, second, counts)
        valid = first_valid & second_valid
        values.extend((first_auc[valid] - second_auc[valid]).tolist())
    values = np.asarray(values[:REPEATS])
    return {
        "mean": float(values.mean()),
        "ci95": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
        "repeats": REPEATS,
    }


def main() -> None:
    information = json.loads((BASE / "information_matched_audit.json").read_text())
    result = {"information_scores": {}, "paired_scores": {}}
    for corpus in ("aider", "hurricane"):
        result["information_scores"][corpus] = {}
        for seed in (101, 202, 303):
            path = BASE / f"information_scores/{corpus}_seed_{seed}.npz"
            arrays = np.load(path)
            errors = arrays["errors"]
            metrics = {
                score: float(roc_auc_score(errors, arrays[score]))
                for score in (
                    "confidence",
                    "maxlogit",
                    "energy",
                    "eo_knn",
                    "vim",
                    "received_consistency",
                    "privileged_consistency",
                    "fine_reference_consistency",
                )
            }
            for score, value in metrics.items():
                expected = (
                    information[corpus]["seeds"][str(seed)]["metrics"][score][
                        "auroc"
                    ]
                    if score
                    in information[corpus]["seeds"][str(seed)]["metrics"]
                    else None
                )
                if expected is not None and not np.isclose(value, expected):
                    raise AssertionError((corpus, seed, score, value, expected))
            result["information_scores"][corpus][str(seed)] = {
                "n": int(len(errors)),
                "n_errors": int(errors.sum()),
                "auroc": metrics,
                "vim_minus_eo_knn": paired_bootstrap(
                    errors,
                    arrays["vim"],
                    arrays["eo_knn"],
                    270_000 + seed,
                ),
            }

    paired = json.loads(
        (BASE / "crasar/leakage_free_paired_gsd_v2.json").read_text()
    )
    index = json.loads(
        (BASE / "crasar/paired_scores/paired_score_index.json").read_text()
    )
    if any(
        str(value).startswith("/")
        for row in index
        for value in row.values()
    ):
        raise AssertionError("portable paired score index contains absolute path")
    sites = np.asarray([row["site"] for row in index])
    for seed in (101, 202, 303):
        arrays = np.load(BASE / f"crasar/paired_scores/seed_{seed}.npz")
        labels = arrays["labels"]
        transitions = {
            site: {
                "n": int(np.sum(sites == site)),
                "uas_correct_satellite_wrong": int(
                    np.sum(
                        (sites == site)
                        & (arrays["uas_prediction"] == labels)
                        & (arrays["satellite_prediction"] != labels)
                    )
                ),
                "uas_wrong_satellite_correct": int(
                    np.sum(
                        (sites == site)
                        & (arrays["uas_prediction"] != labels)
                        & (arrays["satellite_prediction"] == labels)
                    )
                ),
            }
            for site in sorted(set(sites))
        }
        satellite_errors = (arrays["satellite_prediction"] != labels).astype(int)
        confidence_auc = float(
            roc_auc_score(satellite_errors, 1.0 - arrays["satellite_confidence"])
        )
        rcg_auc = float(roc_auc_score(satellite_errors, arrays["satellite_rcg"]))
        expected = paired["seeds"][str(seed)]["pooled"]["satellite"]
        if not np.isclose(confidence_auc, expected["auroc_confidence"]):
            raise AssertionError(("confidence", seed))
        if not np.isclose(rcg_auc, expected["auroc_rcg"]):
            raise AssertionError(("rcg", seed))
        result["paired_scores"][str(seed)] = {
            "n": int(len(labels)),
            "satellite_confidence_auroc": confidence_auc,
            "satellite_rcg_auroc": rcg_auc,
            "prediction_transitions": transitions,
        }
    result["site_specific_intervals"] = {
        seed: {
            site: values["spatial_cluster_bootstrap"]["satellite_auroc_lift"]
            for site, values in seed_result["sites"].items()
        }
        for seed, seed_result in paired["seeds"].items()
    }
    result["all_pass"] = True
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print("PER_EXAMPLE_OUTPUTS_VERIFIED", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
