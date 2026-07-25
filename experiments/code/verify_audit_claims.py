#!/usr/bin/env python3
"""Machine-check all headline audit claims."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/imported/experiments/derived/greatness_iteration3"


def close(actual: float, expected: float, tolerance: float = 5e-4) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=0):
        raise AssertionError(f"{actual} != {expected}")


def main() -> None:
    information = json.loads((BASE / "information_matched_audit.json").read_text())
    aider = json.loads((BASE / "aider_dedup_audit.json").read_text())
    hurricane = json.loads((BASE / "hurricane_dedup_audit.json").read_text())
    crasar = json.loads(
        (BASE / "crasar/leakage_free_crasar.json").read_text()
    )
    paired = json.loads(
        (BASE / "crasar/leakage_free_paired_gsd.json").read_text()
    )
    adjudication = json.loads(
        (
            ROOT / "experiments/derived/paired_protocol_adjudication.json"
        ).read_text()
    )
    checks = {}
    checks["hash_counts"] = (
        len(aider["conflicting_groups"]) == 2
        and len(aider["excluded_paths"]) == 4
        and len(hurricane["cross_split_groups"]) == 3
        and len(hurricane["excluded_indexes"]) == 6
        and not aider["remaining_cross_split_hashes"]
        and not hurricane["remaining_cross_split_hashes"]
    )
    close(
        information["aider"]["aggregate"]["scores"][
            "privileged_consistency"
        ]["mean_auroc"],
        0.964,
    )
    close(
        information["aider"]["aggregate"]["scores"][
            "received_consistency"
        ]["mean_auroc"],
        0.623,
    )
    close(
        information["hurricane"]["aggregate"]["scores"][
            "privileged_consistency"
        ]["mean_auroc"],
        0.989,
    )
    close(
        information["hurricane"]["aggregate"]["scores"][
            "received_consistency"
        ]["mean_auroc"],
        0.590,
    )
    checks["a1"] = information["a1_privileged_inflation_pass"]
    checks["a2"] = information["a2_matched_baseline_complete"]
    checks["crasar_zero_overlap"] = (
        crasar["split_audit"]["cross_split_intersections"] == []
    )
    checks["crasar_internal_valid"] = crasar["aggregate"]["w11_internal_pass"]
    checks["pair_count"] = (
        sum(row["n"] for row in paired["pair_counts"].values()) == 1441
    )
    checks["four_sites_two_events"] = (
        len(paired["pair_counts"]) == 4
        and len({site["event"] for site in paired["protocol"]["sites"]}) == 2
    )
    checks["w11"] = paired["aggregate"]["criteria"]["w11_classifier_validity"]
    checks["w12_failed"] = not paired["aggregate"]["criteria"][
        "w12_deployable_ranking"
    ]
    checks["w13_preregistered_pass"] = adjudication["criteria"][
        "w13_preregistered_mean_rule_pass"
    ]
    checks["w13_all_seed_sensitivity_failed"] = not adjudication["criteria"][
        "w13_posthoc_all_seed_robustness_pass"
    ]
    site_lifts = paired["aggregate"]["site_mean_satellite_auroc_lifts"]
    checks["event_reversal"] = (
        site_lifts["harlem-heights"] > 0
        and site_lifts["mcgregor-college-parkway-south-1"] > 0
        and site_lifts["mexico-beach-2018-10-13"] < 0
        and site_lifts["mexico-beach-2018-10-14"] < 0
    )
    checks["cluster_intervals_include_zero"] = all(
        result["spatial_cluster_bootstrap"]["satellite_auroc_lift"]["ci95"][0]
        <= 0
        <= result["spatial_cluster_bootstrap"]["satellite_auroc_lift"]["ci95"][1]
        for result in paired["seeds"].values()
    )
    if not all(checks.values()):
        raise AssertionError({key: value for key, value in checks.items() if not value})
    output = ROOT / "experiments/derived/claim_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checks": checks, "all_pass": True}, indent=2) + "\n")
    print("AUDIT_CLAIMS_VERIFIED", len(checks), output, flush=True)


if __name__ == "__main__":
    main()
