#!/usr/bin/env python3
"""Degraded-scale RCG analysis and from-scratch training seed."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import run_aider_resolution as base


def analyze(model, splits, device, out_path: Path):
    ladder = base.evaluate_ladder(model, splits["test"], device)
    d_hard, d_js, _, _ = base.disagreement_scores(ladder)
    rows = {}
    for s, res in ladder.items():
        err = (res["pred"] != res["ys"]).astype(int)
        a_d = base.auroc_safe(err, d_hard)
        a_c = base.auroc_safe(err, 1.0 - res["conf"])
        a_js = base.auroc_safe(err, d_js)
        is_fc = (res["ys"] == base.NORMAL) & np.isin(res["pred"], list(base.DISASTER))
        matched = base.match_coverage_fcr(
            base.selective_curves(1.0 - d_hard, is_fc),
            base.selective_curves(res["conf"], is_fc),
        )
        rows[s] = {
            "acc": res["accuracy"],
            "mean_conf": res["mean_confidence"],
            "err_conf": res["mean_confidence_on_errors"],
            "fcr": res["false_critical_rate"],
            "auroc_1mconf": a_c,
            "auroc_d": a_d,
            "auroc_js": a_js,
            "lift_d_minus_1mconf": a_d - a_c,
            "matched_~0.7": matched["at_coverage_~0.7"],
        }
    out_path.write_text(json.dumps({"ladder_summary": base.strip_arrays(ladder), "per_scale": rows}, indent=2))
    print(json.dumps(rows, indent=2))
    return rows


def main():
    root = Path(__file__).resolve().parents[1]
    splits = json.loads((root / "derived/aider_rcg/aider_splits.json").read_text())
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    obj = torch.load(
        root / "derived/aider_rcg/seed_local/model_seed_local.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = base.build_model()
    model.load_state_dict(obj["state_dict"], strict=False)
    model.to(device)
    print("=== LOCAL CKPT degraded-scale ===")
    analyze(model, splits, device, root / "derived/aider_rcg/seed_local/degraded_scale_analysis.json")

    print("=== TRAIN scratch seed 42 ===")
    ckpt, hist, best = base.train_one(
        splits, root / "derived/aider_rcg/seed_scratch42", 42, device, epochs=10, multi_res=False
    )
    model2 = base.build_model()
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    model2.load_state_dict(st["model"])
    model2.to(device)
    print("best_val_f1", best)
    rows = analyze(model2, splits, device, root / "derived/aider_rcg/seed_scratch42/degraded_scale_analysis.json")
    ladder = base.evaluate_ladder(model2, splits["test"], device)
    d_hard, _, errors, baser = base.disagreement_scores(ladder)
    is_fc = (baser["ys"] == base.NORMAL) & np.isin(baser["pred"], list(base.DISASTER))
    out = {
        "best_val_f1": best,
        "history": hist,
        "native_auroc_1mconf": base.auroc_safe(errors, 1 - baser["conf"]),
        "native_auroc_d": base.auroc_safe(errors, d_hard),
        "native_lift": base.auroc_safe(errors, d_hard) - base.auroc_safe(errors, 1 - baser["conf"]),
        "matched_native": base.match_coverage_fcr(
            base.selective_curves(1 - d_hard, is_fc), base.selective_curves(baser["conf"], is_fc)
        )["at_coverage_~0.7"],
        "per_scale": rows,
        "ladder": base.strip_arrays(ladder),
    }
    (root / "derived/aider_rcg/seed_scratch42/metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in out if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
