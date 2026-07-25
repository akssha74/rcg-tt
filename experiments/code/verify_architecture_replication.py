#!/usr/bin/env python3
"""Verify architecture/operator replication from summaries and raw arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/derived/architecture_replication"


def main() -> None:
    mobile = json.loads((BASE / "mobilenet_aider_operator_audit.json").read_text())
    resnet = json.loads((BASE / "resnet_operator_sensitivity.json").read_text())
    checks = {
        "m1_architecture_pass": mobile["aggregate"]["m1_architecture_pass"],
        "m2_operator_pass": mobile["aggregate"]["m2_operator_pass"],
        "m3_cross_architecture_operator_pass": resnet["aggregate"][
            "m3_cross_architecture_operator_pass"
        ],
    }
    raw_checks = 0
    for seed, seed_result in mobile["seeds"].items():
        for operator, summary in seed_result["operators"].items():
            arrays = np.load(ROOT / summary["score_path"])
            errors = arrays["errors"]
            fine_auc = roc_auc_score(
                errors, arrays["fine_reference_consistency"]
            )
            received_auc = roc_auc_score(
                errors, arrays["received_consistency"]
            )
            if not np.isclose(
                fine_auc, summary["fine_reference_consistency"]["auroc"]
            ):
                raise AssertionError(("mobile-fine", seed, operator))
            if not np.isclose(
                received_auc, summary["received_consistency"]["auroc"]
            ):
                raise AssertionError(("mobile-received", seed, operator))
            raw_checks += 2
    for corpus, corpus_results in resnet["results"].items():
        for seed, seed_result in corpus_results.items():
            for operator, summary in seed_result.items():
                arrays = np.load(ROOT / summary["score_path"])
                errors = arrays["errors"]
                fine_auc = roc_auc_score(
                    errors, arrays["fine_reference_consistency"]
                )
                received_auc = roc_auc_score(
                    errors, arrays["received_consistency"]
                )
                if not np.isclose(
                    fine_auc, summary["fine_reference_consistency"]["auroc"]
                ):
                    raise AssertionError(("resnet-fine", corpus, seed, operator))
                if not np.isclose(
                    received_auc, summary["received_consistency"]["auroc"]
                ):
                    raise AssertionError(
                        ("resnet-received", corpus, seed, operator)
                    )
                raw_checks += 2
    checks["raw_auroc_checks"] = raw_checks
    checks["all_36_intervals_positive"] = all(
        seed_result[operator]["fine_minus_received"]["ci95"][0] > 0
        for corpus_results in resnet["results"].values()
        for seed_result in corpus_results.values()
        for operator in seed_result
    ) and all(
        seed_result["operators"][operator]["fine_minus_received"]["ci95"][0] > 0
        for seed_result in mobile["seeds"].values()
        for operator in seed_result["operators"]
    )
    if not all(
        value if isinstance(value, bool) else value == 72
        for value in checks.values()
    ):
        raise AssertionError(checks)
    result = {"checks": checks, "all_pass": True}
    output = BASE / "verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("ARCHITECTURE_REPLICATION_VERIFIED", output, flush=True)


if __name__ == "__main__":
    main()
