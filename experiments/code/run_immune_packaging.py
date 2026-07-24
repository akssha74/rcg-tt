#!/usr/bin/env python3
"""Package robustness-audit JSON into paper tables/figures (deterministic).

Heavy experiment JSON under experiments/derived/immune/ must already exist.
This script only regenerates LaTeX tables and PDF figures from those files,
and prints SHA-256 digests for the artifact ledger.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/derived/immune"
PAPER_DIRS = (ROOT / "paper",)
if (ROOT / "paper-ijrs").exists():
    PAPER_DIRS += (ROOT / "paper-ijrs",)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def write_table(name: str, content: str) -> list[Path]:
    paths = []
    for paper_dir in PAPER_DIRS:
        path = paper_dir / "tables" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        paths.append(path)
    return paths


def save_figure(fig, name: str) -> list[Path]:
    paths = []
    for paper_dir in PAPER_DIRS:
        figure_dir = paper_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        path = figure_dir / f"{name}.pdf"
        fig.savefig(
            path,
            bbox_inches="tight",
            metadata={"CreationDate": None, "ModDate": None},
        )
        fig.savefig(figure_dir / f"{name}.png", dpi=600, bbox_inches="tight")
        paths.append(path)
    return paths


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    aider = json.loads((OUT / "aider_immune.json").read_text())
    hurr = json.loads((OUT / "hurricane_immune.json").read_text())
    scr = json.loads((OUT / "scratch_seed_immune.json").read_text())
    # Confirmatory (pairwise-native) RCG-JS AUROC at s=8, reported alongside the
    # mean-posterior robustness estimator in the strong-baseline table.
    conf = json.loads(
        (ROOT / "experiments/derived/aider_rcg/confirmatory_summary.json").read_text()
    )
    rcg_js_pairwise = conf["local_ckpt"]["scales"]["8"]["auroc_js"]

    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Operator & Corpus & Acc at $s{=}8$ & AUROC$_{JS}$ & AUROC$_{MSP}$ & Lift \\",
        r"\midrule",
    ]
    for mode, v in aider["multi_degradation"].items():
        lines.append(
            f"{mode} & AIDER & {v['acc_s8']:.3f} & {v['auroc_js']:.3f} & "
            f"{v['auroc_msp']:.3f} & {v['lift']:.3f}\\\\"
        )
    for mode, v in hurr["multi_degradation"].items():
        lines.append(
            f"{mode} & Hurricane & {v['acc']:.3f} & {v['auroc_js']:.3f} & "
            f"{v['auroc_msp']:.3f} & {v['lift']:.3f}\\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    multi_table_paths = write_table("multi_degradation.tex", "\n".join(lines) + "\n")

    b = aider["baselines"]
    blines = [
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Error score at $s{=}8$ (AIDER MobileNet) & AUROC \\",
        r"\midrule",
        rf"$1{{-}}$max-softmax & {b['msp']:.3f} \\",
        rf"Temp-scaled MSP & {b['msp_temp_scaled']:.3f} \\",
        rf"Max-logit & {b['maxlogit']:.3f} \\",
        rf"Energy score & {b['energy']:.3f} \\",
        rf"3-seed ensemble MSP & {b['ensemble_msp']:.3f} \\",
        rf"3-seed ensemble disagreement & {b['ensemble_disagreement']:.3f} \\",
        rf"RCG-JS pairwise-native (confirmatory) & {rcg_js_pairwise:.3f} \\",
        rf"RCG-JS mean-posterior (robustness) & {b['rcg_js']:.3f} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    text = "\n".join(blines) + "\n"
    baseline_table_paths = write_table("baselines_s8.tex", text)

    glines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Corpus & $\gamma$ & Conf cov at $s{=}8$ & RCG cov at $s{=}8$ & Cov gap \\",
        r"\midrule",
    ]
    for g, v in aider["threshold_transfer_multi_gamma"].items():
        s8 = v["s8"]
        gap = s8["js_cov"] - s8["conf_cov"]
        glines.append(
            f"AIDER & {g} & {s8['conf_cov']:.3f} & {s8['js_cov']:.3f} & {gap:.3f}\\\\"
        )
    for g, v in hurr["tt_multi_gamma"].items():
        glines.append(
            f"Hurricane & {g} & {v['conf_cov']:.3f} & {v['js_cov']:.3f} & {v['cov_gap']:.3f}\\\\"
        )
    glines += [r"\bottomrule", r"\end{tabular}"]
    gamma_table_paths = write_table("tt_multi_gamma.tex", "\n".join(glines) + "\n")

    slines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Checkpoint & AUROC lift & FCR rel.\ red.\ at $0.7$ & Conf cov TT & RCG cov TT \\",
        r"\midrule",
    ]
    for name, row in scr["seeds"].items():
        s8 = row["scales"]["8"]
        slines.append(
            f"{name.replace('_', ' ')} & {row['s8_lift']:.3f} & {row['fcr_rel']:.3f} & "
            f"{s8['conf_cov']:.3f} & {s8['js_cov']:.3f}\\\\"
        )
    slines += [r"\bottomrule", r"\end{tabular}"]
    scratch_table_paths = write_table("scratch_seed_tt.tex", "\n".join(slines) + "\n")

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    modes = list(aider["multi_degradation"].keys())
    x = np.arange(len(modes))
    ax.bar(
        x - 0.18,
        [aider["multi_degradation"][m]["lift"] for m in modes],
        width=0.36,
        label="AIDER",
        color="#1f4e79",
    )
    hvals = [
        hurr["multi_degradation"][m]["lift"] if m in hurr["multi_degradation"] else np.nan
        for m in modes
    ]
    ax.bar(x + 0.18, hvals, width=0.36, label="Hurricane", color="#c45c26")
    ax.axhline(0.05, color="gray", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15)
    ax.set_ylabel("AUROC lift (JS - MSP)")
    ax.legend(frameon=False)
    fig.tight_layout()
    multi_figure_paths = save_figure(fig, "multi_degradation")
    plt.close()

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    gs = sorted(aider["threshold_transfer_multi_gamma"].keys(), key=float)
    ax.plot(
        [float(g) for g in gs],
        [aider["threshold_transfer_multi_gamma"][g]["s8"]["js_cov"] for g in gs],
        "-o",
        label="AIDER RCG",
        color="#1f4e79",
    )
    ax.plot(
        [float(g) for g in gs],
        [aider["threshold_transfer_multi_gamma"][g]["s8"]["conf_cov"] for g in gs],
        "--s",
        label="AIDER conf",
        color="#1f4e79",
        alpha=0.5,
    )
    ax.plot(
        [float(g) for g in gs],
        [hurr["tt_multi_gamma"][g]["js_cov"] for g in gs],
        "-o",
        label="Hurricane RCG",
        color="#c45c26",
    )
    ax.plot(
        [float(g) for g in gs],
        [hurr["tt_multi_gamma"][g]["conf_cov"] for g in gs],
        "--s",
        label="Hurricane conf",
        color="#c45c26",
        alpha=0.5,
    )
    ax.set_xlabel(r"Native coverage target $\gamma$")
    ax.set_ylabel("Test coverage at $s=8$")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    gamma_figure_paths = save_figure(fig, "tt_multi_gamma")
    plt.close()

    paths = [
        OUT / "aider_immune.json",
        OUT / "hurricane_immune.json",
        OUT / "scratch_seed_immune.json",
        multi_table_paths[0],
        baseline_table_paths[0],
        gamma_table_paths[0],
        scratch_table_paths[0],
        multi_figure_paths[0],
        gamma_figure_paths[0],
    ]
    for p in paths:
        print(f"{p.relative_to(ROOT)} {sha256(p)}")


if __name__ == "__main__":
    main()
