#!/usr/bin/env python3
"""Adjudicate W11-W13 exactly against the frozen corrective preregistration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "experiments/imported/experiments/derived/greatness_iteration3/"
    "crasar/leakage_free_paired_gsd.json"
)
OUTPUT = ROOT / "experiments/derived/paired_protocol_adjudication.json"
SEEDS = ("101", "202", "303")


def mean(values):
    return float(np.mean(np.asarray(values, dtype=float)))


def main() -> None:
    data = json.loads(SOURCE.read_text())
    uas_accuracy = [
        data["seeds"][seed]["pooled"]["uas"]["accuracy"] for seed in SEEDS
    ]
    uas_majority = [
        data["seeds"][seed]["pooled"]["uas"]["majority_accuracy"]
        for seed in SEEDS
    ]
    uas_balanced = [
        data["seeds"][seed]["pooled"]["uas"]["balanced_accuracy"]
        for seed in SEEDS
    ]
    transfer = {
        gate: {
            metric: [
                data["seeds"][seed]["pooled"]["satellite_transfer"]["0.7"][
                    gate
                ][metric]
                for seed in SEEDS
            ]
            for metric in ("absolute_coverage_error", "selective_risk")
        }
        for gate in ("confidence", "rcg")
    }
    mean_transfer = {
        gate: {metric: mean(values) for metric, values in metrics.items()}
        for gate, metrics in transfer.items()
    }
    w11 = (
        all(value > baseline for value, baseline in zip(uas_accuracy, uas_majority))
        and mean(uas_balanced) >= 0.60
    )
    site_lifts = data["aggregate"]["site_mean_satellite_auroc_lifts"]
    w12 = (
        data["aggregate"]["mean_satellite_auroc_lift"] > 0
        and sum(value > 0 for value in site_lifts.values()) >= 3
        and all(
            data["seeds"][seed]["spatial_cluster_bootstrap"][
                "satellite_auroc_lift"
            ]["ci95"][0]
            > 0
            for seed in SEEDS
        )
    )
    w13_preregistered = (
        mean_transfer["rcg"]["absolute_coverage_error"]
        <= mean_transfer["confidence"]["absolute_coverage_error"]
        and mean_transfer["rcg"]["selective_risk"]
        <= mean_transfer["confidence"]["selective_risk"]
    )
    w13_all_seed_sensitivity = all(
        transfer["rcg"]["absolute_coverage_error"][index]
        <= transfer["confidence"]["absolute_coverage_error"][index]
        and transfer["rcg"]["selective_risk"][index]
        <= transfer["confidence"]["selective_risk"][index]
        for index in range(len(SEEDS))
    )
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "criteria": {
            "w11_preregistered_pass": w11,
            "w12_preregistered_pass": w12,
            "w13_preregistered_mean_rule_pass": w13_preregistered,
            "w13_posthoc_all_seed_robustness_pass": w13_all_seed_sensitivity,
        },
        "w13_values": {
            "confidence": {
                **mean_transfer["confidence"],
                "seed_values": transfer["confidence"],
            },
            "received_consistency": {
                **mean_transfer["rcg"],
                "seed_values": transfer["rcg"],
            },
        },
        "deviation": {
            "original_result_json_field": (
                "aggregate.criteria.w13_threshold_transfer applies an "
                "all-seed rule not present in the frozen mean-based W13"
            ),
            "manuscript_resolution": (
                "Report preregistered W13 as passed and the all-seed result "
                "as a failed post-hoc robustness sensitivity."
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print("PAIRED_PROTOCOL_ADJUDICATED", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
