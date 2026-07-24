#!/usr/bin/env python3
"""Generate multi-seed, EO-OOD, and measured-GSD paper artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "experiments/derived/greatness_strengthening"
PAPER_DIRS = [ROOT / "paper"]
if (ROOT / "paper-ijrs").exists():
    PAPER_DIRS.append(ROOT / "paper-ijrs")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_sd(metric: dict) -> str:
    return f"{metric['mean']:.3f}$\\pm${metric['std']:.3f}"


def write_all(relative: str, content: str) -> list[Path]:
    paths = []
    for paper in PAPER_DIRS:
        path = paper / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        paths.append(path)
    return paths


def save_all(name: str, fig) -> list[Path]:
    paths = []
    for paper in PAPER_DIRS:
        directory = paper / "figures"
        directory.mkdir(parents=True, exist_ok=True)
        pdf = directory / f"{name}.pdf"
        png = directory / f"{name}.png"
        fig.savefig(
            pdf,
            bbox_inches="tight",
            metadata={"CreationDate": None, "ModDate": None},
        )
        fig.savefig(png, dpi=600, bbox_inches="tight")
        paths.append(pdf)
    return paths


def main() -> None:
    primary_path = DERIVED / "primary_multiseed.json"
    measured_path = DERIVED / "crasar/measured_gsd_crasar.json"
    paired_path = DERIVED / "crasar/paired_measured_gsd.json"
    multisite_path = DERIVED / "crasar/multisite_paired_gsd.json"
    primary = json.loads(primary_path.read_text())
    measured = json.loads(measured_path.read_text())
    paired = json.loads(paired_path.read_text())
    multisite = json.loads(multisite_path.read_text())

    table = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Corpus & AUROC$_{1-c}$ & AUROC$_{kNN}$ & AUROC$_{RCG}$ & "
        r"RCG$-(1-c)$ & RCG$-kNN$ \\",
        r"\midrule",
    ]
    labels = {"aider": "AIDER ResNet-18", "hurricane": "Hurricane ResNet-18"}
    for key in ("aider", "hurricane"):
        aggregate = primary[key]["aggregate"]
        table.append(
            f"{labels[key]} & "
            f"{mean_sd(aggregate['auroc_confidence'])} & "
            f"{mean_sd(aggregate['auroc_knn'])} & "
            f"{mean_sd(aggregate['auroc_rcg'])} & "
            f"{mean_sd(aggregate['lift_rcg_minus_confidence'])} & "
            f"{mean_sd(aggregate['lift_rcg_minus_knn'])} \\\\"
        )
    table.extend([r"\bottomrule", r"\end{tabular}"])
    primary_tables = write_all(
        "tables/primary_multiseed.tex", "\n".join(table) + "\n"
    )

    measured_table = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Measured GSD & Accuracy & Error conf. & AUROC$_{1-c}$ & "
        r"AUROC$_{RCG}$ & Lift \\",
        r"\midrule",
    ]
    paired_aggregate = multisite["aggregate"]["pooled"]
    measured_rows = [
        ("UAS 3.84--4.67", paired_aggregate["uas"]),
        (
            "Satellite 30.52",
            paired_aggregate["satellite"],
        ),
    ]
    for label, row in measured_rows:
        measured_table.append(
            f"{label}\\,cm\\,px$^{{-1}}$ & "
            f"{mean_sd(row['accuracy'])} & "
            f"{mean_sd(row['error_confidence'])} & "
            f"{mean_sd(row['auroc_confidence'])} & "
            f"{mean_sd(row['auroc_rcg'])} & "
            f"{mean_sd(row['lift_rcg_minus_confidence'])} \\\\"
        )
    measured_table.extend([r"\bottomrule", r"\end{tabular}"])
    measured_tables = write_all(
        "tables/measured_gsd.tex", "\n".join(measured_table) + "\n"
    )

    strata = measured["aggregate"]["strata"]
    unpaired_table = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Measured GSD & Accuracy & Error conf. & AUROC$_{1-c}$ & "
        r"AUROC$_{RCG}$ & Lift \\",
        r"\midrule",
    ]
    for key in sorted(strata, key=float):
        row = strata[key]
        unpaired_table.append(
            f"{float(key) * 100:.2f}\\,cm\\,px$^{{-1}}$ & "
            f"{mean_sd(row['accuracy'])} & "
            f"{mean_sd(row['error_confidence'])} & "
            f"{mean_sd(row['auroc_confidence'])} & "
            f"{mean_sd(row['auroc_rcg'])} & "
            f"{mean_sd(row['lift_rcg_minus_confidence'])} \\\\"
        )
    unpaired_table.extend([r"\bottomrule", r"\end{tabular}"])
    unpaired_tables = write_all(
        "tables/measured_gsd_unpaired.tex",
        "\n".join(unpaired_table) + "\n",
    )

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    x = np.arange(3)
    width = 0.18
    for group, key in enumerate(("aider", "hurricane")):
        seeds = primary[key]["seeds"]
        confidence = [
            seeds[str(seed)]["s8"]["lift_rcg_minus_confidence"]
            for seed in primary["protocol"]["seeds"]
        ]
        knn = [
            seeds[str(seed)]["s8"]["lift_rcg_minus_knn"]
            for seed in primary["protocol"]["seeds"]
        ]
        offset = (-1.5 + 2 * group) * width
        axes[0].bar(
            x + offset,
            confidence,
            width,
            label=f"{labels[key]} vs confidence",
        )
        axes[0].bar(
            x + offset + width,
            knn,
            width,
            alpha=0.55,
            label=f"{labels[key]} vs kNN",
        )
    axes[0].axhline(0, color="0.4", linewidth=0.8)
    axes[0].set_xticks(x, [str(seed) for seed in primary["protocol"]["seeds"]])
    axes[0].set_xlabel("Training seed")
    axes[0].set_ylabel("RCG AUROC lift")
    axes[0].legend(frameon=False, fontsize=6)
    axes[0].text(
        0.02, 0.96, "(a)", transform=axes[0].transAxes, va="top", fontweight="bold"
    )

    accuracy = np.asarray([row[1]["accuracy"]["mean"] for row in measured_rows])
    lift = np.asarray(
        [row[1]["lift_rcg_minus_confidence"]["mean"] for row in measured_rows]
    )
    modality_x = np.arange(2)
    axes[1].bar(modality_x - 0.18, accuracy, width=0.36, label="Accuracy")
    axes[1].bar(modality_x + 0.18, lift, width=0.36, label="RCG AUROC lift")
    axes[1].axhline(0, color="0.4", linewidth=0.8)
    axes[1].set_xticks(
        modality_x,
        [r"UAS 3.84--4.67", r"Satellite 30.52"],
        rotation=8,
    )
    axes[1].set_xlabel(r"Measured GSD (cm px$^{-1}$)")
    axes[1].set_ylabel("Mean across seeds")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].text(
        0.02, 0.96, "(b)", transform=axes[1].transAxes, va="top", fontweight="bold"
    )
    figure.tight_layout()
    figures = save_all("greatness_strengthening", figure)
    plt.close(figure)

    summary = {
        "primary_multiseed": {
            key: {
                "w5_pass": primary[key]["aggregate"]["w5_pass"],
                "w7_pass": primary[key]["aggregate"]["w7_pass"],
                "rcg_minus_confidence": primary[key]["aggregate"][
                    "lift_rcg_minus_confidence"
                ],
                "rcg_minus_knn": primary[key]["aggregate"][
                    "lift_rcg_minus_knn"
                ],
            }
            for key in ("aider", "hurricane")
        },
        "measured_gsd": {
            "initial_unpaired": measured["aggregate"],
            "paired_single_site": paired["aggregate"],
            "paired_multisite": multisite["aggregate"],
        },
        "all_w5_pass": all(
            primary[key]["aggregate"]["w5_pass"] for key in ("aider", "hurricane")
        ),
        "all_w7_pass": all(
            primary[key]["aggregate"]["w7_pass"] for key in ("aider", "hurricane")
        ),
        "w6_initial_unpaired_pass": measured["aggregate"]["w6_pass"],
        "w6b_paired_pass": paired["aggregate"]["w6b_pass"],
        "w6c_multisite_pass": multisite["aggregate"]["w6c_pass"],
        "w6_pass": multisite["aggregate"]["w6c_pass"],
        "artifacts": {
            "primary_table": {
                "path": str(primary_tables[0].relative_to(ROOT)),
                "sha256": sha256(primary_tables[0]),
            },
            "measured_table": {
                "path": str(measured_tables[0].relative_to(ROOT)),
                "sha256": sha256(measured_tables[0]),
            },
            "measured_unpaired_table": {
                "path": str(unpaired_tables[0].relative_to(ROOT)),
                "sha256": sha256(unpaired_tables[0]),
            },
            "figure": {
                "path": str(figures[0].relative_to(ROOT)),
                "sha256": sha256(figures[0]),
            },
        },
    }
    output = DERIVED / "strengthening_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
