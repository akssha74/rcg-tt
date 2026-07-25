#!/usr/bin/env python3
"""Verify third-event Idalia sensitivity from portable per-example arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/derived/idalia_paired"


def main() -> None:
    summary = json.loads((BASE / "idalia_paired_sensitivity.json").read_text())
    checks = {
        "selection_count": summary["selection"]["same_eligible_label"] == 458,
        "joint_clusters": all(
            result["cluster_bootstrap"]["clusters"] == 37
            for result in summary["results"].values()
        ),
        "e1": summary["aggregate"]["e1_classifier_validity"],
        "e2": summary["aggregate"]["e2_completeness"],
    }
    for seed, result in summary["results"].items():
        arrays = np.load(BASE / f"seed_{seed}.npz")
        labels = arrays["labels"]
        uas_balanced = balanced_accuracy_score(
            labels, arrays["uas_prediction"]
        )
        crewed_errors = (arrays["crewed_prediction"] != labels).astype(int)
        confidence_auc = roc_auc_score(
            crewed_errors, 1.0 - arrays["crewed_confidence"]
        )
        rcg_auc = roc_auc_score(crewed_errors, arrays["crewed_rcg"])
        if not np.isclose(uas_balanced, result["uas"]["balanced_accuracy"]):
            raise AssertionError(("uas-balanced", seed))
        if not np.isclose(
            confidence_auc, result["crewed"]["auroc_confidence"]
        ):
            raise AssertionError(("confidence", seed))
        if not np.isclose(rcg_auc, result["crewed"]["auroc_rcg"]):
            raise AssertionError(("rcg", seed))
        checks[f"positive_cluster_interval_{seed}"] = (
            result["cluster_bootstrap"]["ci95"][0] > 0
        )
    if not all(checks.values()):
        raise AssertionError(checks)
    output = BASE / "verification.json"
    output.write_text(json.dumps({"checks": checks, "all_pass": True}, indent=2) + "\n")
    print("IDALIA_SENSITIVITY_VERIFIED", output, flush=True)


if __name__ == "__main__":
    main()
