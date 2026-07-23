#!/usr/bin/env python3
"""Generate paper tables/figures from confirmatory AIDER RCG results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DER = ROOT / "experiments/derived/aider_rcg"
FIG = ROOT / "paper/figures"
TAB = ROOT / "paper/tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main():
    conf = json.loads((DER / "confirmatory_summary.json").read_text())
    local = conf["local_ckpt"]["scales"]
    # trust-collapse figure for local ckpt
    scales = [1, 2, 4, 8]
    acc = [local[str(s)]["acc"] for s in scales]
    errc = [local[str(s)]["err_conf"] for s in scales]
    fcr = [local[str(s)]["fcr"] for s in scales]
    lift = [local[str(s)]["lift_js_minus_1mconf"] for s in scales]

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    ax[0].plot(scales, acc, "o-", label="Accuracy")
    ax[0].plot(scales, errc, "s--", label="Mean conf on errors")
    ax[0].set_xlabel("Downsample factor s")
    ax[0].set_ylabel("Rate")
    ax[0].set_title("Trust collapse (local CNN)")
    ax[0].legend(fontsize=8)
    ax[0].set_xticks(scales)

    ax[1].plot(scales, fcr, "o-", color="crimson")
    ax[1].set_xlabel("Downsample factor s")
    ax[1].set_ylabel("False-critical rate")
    ax[1].set_title("False critical vs resolution")
    ax[1].set_xticks(scales)

    ax[2].bar([str(s) for s in scales], lift, color=["#4c72b0" if v >= 0 else "#c44e52" for v in lift])
    ax[2].axhline(0.05, color="gray", ls=":", label="W1 bar (0.05)")
    ax[2].set_xlabel("Operating scale s")
    ax[2].set_ylabel("AUROC lift (JS − (1−conf))")
    ax[2].set_title("RCG-JS vs confidence")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig_path = FIG / "trust_collapse_rcg.pdf"
    fig.savefig(fig_path)
    fig.savefig(FIG / "trust_collapse_rcg.png", dpi=150)
    plt.close()

    # main results table
    def fmt(x, nd=3):
        return f"{x:.{nd}f}"

    rows = []
    for name, key in [
        ("Local CNN", "local_ckpt"),
        ("Scratch seed 42", "scratch_42"),
        ("Scratch seed 43", "scratch_43"),
        ("Scratch seed 44", "scratch_44"),
        ("Multi-res train", "multires_42"),
    ]:
        s8 = conf[key]["scales"]["8"]
        s1 = conf[key]["scales"]["1"]
        rows.append(
            (
                name,
                fmt(s1["acc"]),
                fmt(s8["acc"]),
                fmt(s8["err_conf"]),
                fmt(s8["auroc_1mconf"]),
                fmt(s8["auroc_js"]),
                fmt(s8["lift_js_minus_1mconf"]),
                fmt(s8["matched_js_vs_conf_~0.7"]["relative_reduction_a_vs_b"]),
            )
        )

    tex = []
    tex.append("\\begin{tabular}{lccccccc}")
    tex.append("\\toprule")
    tex.append("Model & Acc $s{=}1$ & Acc $s{=}8$ & Err-conf $s{=}8$ & AUROC$_{1-c}$ & AUROC$_{JS}$ & Lift & Rel.\\ FCR red.\\\\")
    tex.append("\\midrule")
    for r in rows:
        tex.append(" & ".join(r) + "\\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tab_path = TAB / "main_rcg_s8.tex"
    tab_path.write_text("\n".join(tex) + "\n")

    # claim ledger helper values
    claims = {
        "local_s8_lift": local["8"]["lift_js_minus_1mconf"],
        "local_s8_lift_ci": local["8"]["bootstrap_lift_js"]["ci95"],
        "local_s8_fcr_rel": local["8"]["matched_js_vs_conf_~0.7"]["relative_reduction_a_vs_b"],
        "local_acc_ladder": [local[str(s)]["acc"] for s in scales],
        "local_err_conf_ladder": [local[str(s)]["err_conf"] for s in scales],
        "scratch_s8_lift_mean": conf["aggregate_scratch_s8_js_lift"]["mean"],
        "scratch_s8_lift_std": conf["aggregate_scratch_s8_js_lift"]["std"],
        "figure_sha256": sha(fig_path),
        "table_sha256": sha(tab_path),
    }
    (DER / "paper_values.json").write_text(json.dumps(claims, indent=2))
    print(json.dumps(claims, indent=2))


if __name__ == "__main__":
    main()
