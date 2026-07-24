"""Generate cross-corpus accuracy and threshold-transfer figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCALES = ("1", "2", "4", "8")
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


def load_series() -> list[tuple[str, tuple[float, float, float], list[float]]]:
    """Load the three confirmatory accuracy ladders plotted in the paper."""
    with (ROOT / "experiments/derived/aider_rcg/confirmatory_summary.json").open() as f:
        aider = json.load(f)
    with (ROOT / "experiments/derived/aider_rcg/resnet18_confirmatory.json").open() as f:
        aider_resnet = json.load(f)
    with (ROOT / "experiments/derived/hurricane_rcg/confirmatory_summary.json").open() as f:
        hurricane = json.load(f)

    return [
        (
            "AIDER MobileNet",
            (0.08, 0.32, 0.62),
            [aider["local_ckpt"]["ladder"][s]["accuracy"] for s in SCALES],
        ),
        (
            "AIDER ResNet-18",
            (0.85, 0.33, 0.10),
            [aider_resnet["ladder_acc"][s] for s in SCALES],
        ),
        (
            "Hurricane ResNet-18",
            (0.18, 0.55, 0.30),
            [hurricane["scales"][s]["acc"] for s in SCALES],
        ),
    ]


def save_figure(fig, name: str) -> None:
    for paper_dir in PAPER_DIRS:
        figure_dir = paper_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        pdf = figure_dir / f"{name}.pdf"
        png = figure_dir / f"{name}.png"
        fig.savefig(
            pdf,
            bbox_inches="tight",
            metadata={"CreationDate": None, "ModDate": None},
        )
        fig.savefig(png, dpi=600, bbox_inches="tight")
        print(pdf.relative_to(ROOT))


def accuracy_figure() -> None:
    series = load_series()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    styles = (("-", "o"), ("--", "s"), (":", "^"))
    for (label, colour, values), (linestyle, marker) in zip(series, styles):
        ax.plot(
            [int(scale) for scale in SCALES],
            values,
            linestyle=linestyle,
            marker=marker,
            color=colour,
            linewidth=1.8,
            label=label,
        )
    ax.set_ylim(0.6, 1.0)
    ax.set_xticks([int(scale) for scale in SCALES])
    ax.set_xlabel("GSD-proxy factor $s$")
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "accuracy_ladder")
    plt.close(fig)


def threshold_transfer_figure() -> None:
    with (ROOT / "experiments/derived/aider_rcg/threshold_transfer.json").open() as f:
        aider = json.load(f)
    with (ROOT / "experiments/derived/strengthen_unambiguous/summary.json").open() as f:
        hurricane = json.load(f)["hurricane"]["transfer"]

    rows = [
        (
            "AIDER MobileNet",
            [
                aider["transfer_to_test_s1"]["conf"]["coverage"],
                aider["transfer_to_test_s8"]["conf"]["coverage"],
            ],
            [
                aider["transfer_to_test_s1"]["js_rcg"]["coverage"],
                aider["transfer_to_test_s8"]["js_rcg"]["coverage"],
            ],
        ),
        (
            "Hurricane ResNet-18",
            [hurricane["s1"]["conf"]["coverage"], hurricane["s8"]["conf"]["coverage"]],
            [hurricane["s1"]["js"]["coverage"], hurricane["s8"]["js"]["coverage"]],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    x = np.arange(2)
    width = 0.36
    for index, (label, confidence, rcg) in enumerate(rows):
        ax = axes[index]
        ax.bar(x - width / 2, confidence, width, label="Confidence", color="#4c78a8")
        ax.bar(x + width / 2, rcg, width, label="RCG-JS", color="#f58518")
        ax.axhline(0.7, color="0.5", linestyle=":", linewidth=1)
        ax.set_xticks(x, ("$s=1$", "$s=8$"))
        ax.set_xlabel(label)
        ax.set_ylim(0, 1)
        ax.text(0.02, 0.96, f"({chr(97 + index)})", transform=ax.transAxes, va="top", fontweight="bold")
    axes[0].set_ylabel("Auto-decision coverage")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "threshold_transfer")
    plt.close(fig)


def main() -> None:
    accuracy_figure()
    threshold_transfer_figure()


if __name__ == "__main__":
    main()
