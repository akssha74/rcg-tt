#!/usr/bin/env python3
"""Evaluate a saved AIDER MobileNet checkpoint under the RCG resolution protocol."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_aider_resolution as base


def load_model(ckpt_path: Path, device):
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = base.build_model()
    if isinstance(obj, dict) and "model" in obj:
        model.load_state_dict(obj["model"])
        meta = {k: obj[k] for k in obj if k != "model"}
    elif isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
        # strip possible prefixes
        cleaned = {}
        for k, v in sd.items():
            nk = k
            for pref in ("module.", "model."):
                if nk.startswith(pref):
                    nk = nk[len(pref) :]
            cleaned[nk] = v
        model.load_state_dict(cleaned, strict=False)
        meta = {"classes": obj.get("classes")}
    else:
        model.load_state_dict(obj)
        meta = {}
    model.to(device).eval()
    return model, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", type=str, default="eval")
    args = ap.parse_args()

    splits = json.loads(args.splits.read_text())
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, meta = load_model(args.ckpt, device)
    ladder = base.evaluate_ladder(model, splits["test"], device)
    d_hard, d_js, errors, baser = base.disagreement_scores(ladder)
    is_fc = (baser["ys"] == base.NORMAL) & np.isin(baser["pred"], list(base.DISASTER))
    auroc_conf = base.auroc_safe(errors, 1.0 - baser["conf"])
    auroc_d = base.auroc_safe(errors, d_hard)
    auroc_js = base.auroc_safe(errors, d_js)
    diff = base.bootstrap_auroc_diff(errors, d_hard, 1.0 - baser["conf"], seed=0)
    curve_conf = base.selective_curves(baser["conf"], is_fc)
    curve_rcg = base.selective_curves(1.0 - d_hard, is_fc)
    matched = base.match_coverage_fcr(curve_rcg, curve_conf)

    # entropy baseline
    ent = -np.sum(baser["probs"] * np.log(np.clip(baser["probs"], 1e-9, 1)), axis=1)
    auroc_ent = base.auroc_safe(errors, ent)
    curve_ent = base.selective_curves(-ent, is_fc)  # low entropy => keep
    matched_ent = base.match_coverage_fcr(curve_rcg, curve_ent)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out.with_suffix(".npz"),
        d_hard=d_hard,
        d_js=d_js,
        errors=errors,
        conf=baser["conf"],
        ys=baser["ys"],
        pred=baser["pred"],
        is_fc=is_fc.astype(np.uint8),
        probs=baser["probs"],
    )
    row = {
        "tag": args.tag,
        "ckpt": str(args.ckpt),
        "device": str(device),
        "meta": meta,
        "n_test": int(len(baser["ys"])),
        "ladder": base.strip_arrays(ladder),
        "auroc_error_given_one_minus_conf": auroc_conf,
        "auroc_error_given_disagreement": auroc_d,
        "auroc_error_given_js": auroc_js,
        "auroc_error_given_entropy": auroc_ent,
        "auroc_lift_disagreement_minus_one_minus_conf": auroc_d - auroc_conf,
        "bootstrap_auroc_lift": diff,
        "selective_confidence": curve_conf,
        "selective_rcg": curve_rcg,
        "selective_entropy": curve_ent,
        "matched_coverage_rcg_vs_conf": matched["at_coverage_~0.7"],
        "matched_coverage_rcg_vs_entropy": matched_ent["at_coverage_~0.7"],
        "trust_collapse": {
            s: {
                "accuracy": ladder[s]["accuracy"],
                "mean_confidence": ladder[s]["mean_confidence"],
                "mean_confidence_on_errors": ladder[s]["mean_confidence_on_errors"],
                "false_critical_rate": ladder[s]["false_critical_rate"],
            }
            for s in ladder
        },
    }
    args.out.write_text(json.dumps(row, indent=2))
    print(json.dumps({k: row[k] for k in [
        "n_test",
        "auroc_error_given_one_minus_conf",
        "auroc_error_given_disagreement",
        "auroc_lift_disagreement_minus_one_minus_conf",
        "matched_coverage_rcg_vs_conf",
        "trust_collapse",
    ]}, indent=2))


if __name__ == "__main__":
    main()
