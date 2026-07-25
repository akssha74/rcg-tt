#!/usr/bin/env python3
"""Verify reveal/mask correspondence experiment from raw permutation arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/derived/reference_reveal_mask"


def main() -> None:
    summary = json.loads((BASE / "reveal_mask_summary.json").read_text())
    checks = {
        "combinations": summary["aggregate"]["combinations"] == 24,
        "all_pass": summary["aggregate"]["m4_reveal_mask_pass"],
        "minimum_gap_positive": summary["aggregate"]["minimum_gap"] > 0,
        "maximum_probability": (
            summary["aggregate"]["maximum_empirical_probability"] <= 0.05
        ),
    }
    raw_checks = 0
    for corpus, corpus_results in summary["results"].items():
        for seed, seed_results in corpus_results.items():
            for operator, result in seed_results.items():
                arrays = np.load(ROOT / result["score_path"])
                errors = arrays["errors"]
                aligned_auc = roc_auc_score(errors, arrays["aligned_score"])
                masked_aurocs = np.asarray(
                    [
                        roc_auc_score(errors, row)
                        for row in arrays["masked_scores"]
                    ]
                )
                if not np.isclose(aligned_auc, result["aligned_auroc"]):
                    raise AssertionError(("aligned", corpus, seed, operator))
                if not np.allclose(masked_aurocs, arrays["masked_aurocs"]):
                    raise AssertionError(("masked", corpus, seed, operator))
                if not np.isclose(
                    aligned_auc - masked_aurocs.mean(),
                    result["aligned_minus_masked_mean"],
                ):
                    raise AssertionError(("gap", corpus, seed, operator))
                raw_checks += 102
    checks["raw_metric_checks"] = raw_checks
    if not all(
        value if isinstance(value, bool) else value == 2448
        for value in checks.values()
    ):
        raise AssertionError(checks)
    output = BASE / "verification.json"
    output.write_text(json.dumps({"checks": checks, "all_pass": True}, indent=2) + "\n")
    print("REFERENCE_REVEAL_MASK_VERIFIED", output, flush=True)


if __name__ == "__main__":
    main()
