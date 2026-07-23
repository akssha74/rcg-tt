#!/usr/bin/env python3
"""Confirmatory degraded-scale (s=8) RCG-JS vs confidence; multi-seed scratch; TTA baseline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

import run_aider_resolution as base


def js_rcg_eval(model, splits, device):
    ladder = base.evaluate_ladder(model, splits["test"], device)
    d_hard, d_js, _, _ = base.disagreement_scores(ladder)
    out = {"ladder": base.strip_arrays(ladder), "scales": {}}
    for s, res in ladder.items():
        err = (res["pred"] != res["ys"]).astype(int)
        is_fc = (res["ys"] == base.NORMAL) & np.isin(res["pred"], list(base.DISASTER))
        a_c = base.auroc_safe(err, 1.0 - res["conf"])
        a_d = base.auroc_safe(err, d_hard)
        a_js = base.auroc_safe(err, d_js)
        lift_js = base.bootstrap_auroc_diff(err, d_js, 1.0 - res["conf"], seed=0)
        matched_js = base.match_coverage_fcr(
            base.selective_curves(-d_js, is_fc),  # lower JS => keep
            base.selective_curves(res["conf"], is_fc),
        )["at_coverage_~0.7"]
        out["scales"][s] = {
            "acc": res["accuracy"],
            "err_conf": res["mean_confidence_on_errors"],
            "fcr": res["false_critical_rate"],
            "auroc_1mconf": a_c,
            "auroc_d_hard": a_d,
            "auroc_js": a_js,
            "lift_js_minus_1mconf": a_js - a_c,
            "bootstrap_lift_js": lift_js,
            "matched_js_vs_conf_~0.7": matched_js,
        }
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    splits = json.loads((root / "derived/aider_rcg/aider_splits.json").read_text())
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    results = {}

    # local strong ckpt
    obj = torch.load(root / "derived/aider_rcg/seed_local/model_seed_local.pt", map_location="cpu", weights_only=False)
    m = base.build_model(); m.load_state_dict(obj["state_dict"], strict=False); m.to(device)
    results["local_ckpt"] = js_rcg_eval(m, splits, device)

    # scratch seeds
    scratch_lifts = []
    for seed in (42, 43, 44):
        out_dir = root / f"derived/aider_rcg/seed_scratch{seed}"
        ckpt = out_dir / f"model_seed{seed}.pt"
        if not ckpt.exists():
            print("training seed", seed)
            ckpt, hist, best = base.train_one(splits, out_dir, seed, device, epochs=10)
            (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
        else:
            best = None
        m2 = base.build_model()
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        m2.load_state_dict(st["model"]); m2.to(device)
        ev = js_rcg_eval(m2, splits, device)
        ev["best_val_f1"] = best if best is not None else st.get("val_macro_f1")
        results[f"scratch_{seed}"] = ev
        scratch_lifts.append(ev["scales"]["8"]["lift_js_minus_1mconf"])

    results["aggregate_scratch_s8_js_lift"] = {
        "mean": float(np.mean(scratch_lifts)),
        "std": float(np.std(scratch_lifts)),
        "values": scratch_lifts,
    }

    # multi-res curriculum single seed
    out_dir = root / "derived/aider_rcg/seed_multires42"
    ckpt = out_dir / "model_seed42.pt"
    if not ckpt.exists():
        print("training multires")
        ckpt, hist, best = base.train_one(splits, out_dir, 42, device, epochs=10, multi_res=True)
        (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    m3 = base.build_model(); st = torch.load(ckpt, map_location="cpu", weights_only=False)
    m3.load_state_dict(st["model"]); m3.to(device)
    results["multires_42"] = js_rcg_eval(m3, splits, device)

    out = root / "derived/aider_rcg/confirmatory_summary.json"
    out.write_text(json.dumps(results, indent=2))
    # print headline
    headline = {
        "local_s8": results["local_ckpt"]["scales"]["8"],
        "scratch_agg_s8_lift": results["aggregate_scratch_s8_js_lift"],
        "scratch42_s8": results["scratch_42"]["scales"]["8"],
        "multires_s8": results["multires_42"]["scales"]["8"],
    }
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
