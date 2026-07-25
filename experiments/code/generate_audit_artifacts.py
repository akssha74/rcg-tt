#!/usr/bin/env python3
"""Generate information-audit tables and figure from registered JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT / "experiments/imported/experiments/derived/greatness_iteration3"
)
SOURCE = EVIDENCE / "information_matched_audit.json"
AIDER_AUDIT = (
    EVIDENCE / "aider_dedup_audit.json"
)
HURRICANE_AUDIT = (
    EVIDENCE / "hurricane_dedup_audit.json"
)
CRASAR_SOURCE = EVIDENCE / "crasar/leakage_free_crasar.json"
PAIRED_SOURCE = (
    EVIDENCE / "crasar/leakage_free_paired_gsd.json"
)
MOBILENET_SOURCE = (
    ROOT
    / "experiments/derived/architecture_replication/"
    "mobilenet_aider_operator_audit.json"
)
RESNET_OPERATOR_SOURCE = (
    ROOT
    / "experiments/derived/architecture_replication/"
    "resnet_operator_sensitivity.json"
)
IDALIA_SOURCE = (
    ROOT / "experiments/derived/idalia_paired/idalia_paired_sensitivity.json"
)
REVEAL_MASK_SOURCE = (
    ROOT
    / "experiments/derived/reference_reveal_mask/reveal_mask_summary.json"
)
TABLES = ROOT / "paper/tables"
FIGURES = ROOT / "paper/figures"
SCORES = [
    ("confidence", "Confidence"),
    ("maxlogit", "Max logit"),
    ("energy", "Energy"),
    ("eo_knn", "EO-kNN"),
    ("vim", "ViM"),
    ("received_consistency", "Received consistency"),
    ("privileged_consistency", "Reference-dependent diagnostic"),
]
FIGURE_LABELS = ["MSP", "MaxLogit", "Energy", "EO-kNN", "ViM", "Recv-JS", "Ref-JS"]


def write_score_table(data: dict) -> None:
    rows = []
    for key, label in SCORES:
        aider = data["aider"]["aggregate"]["scores"][key]
        hurricane = data["hurricane"]["aggregate"]["scores"][key]
        rows.append(
            f"{label} & {aider['mean_auroc']:.3f} $\\pm$ "
            f"{aider['std_auroc']:.3f} & {hurricane['mean_auroc']:.3f} "
            f"$\\pm$ {hurricane['std_auroc']:.3f} \\\\"
        )
    text = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Error-detection AUROC under matched received-image "
            "information. The final row is a non-deployable diagnostic with "
            "access to the retained finer reference. Values are mean $\\pm$ "
            "sample SD across three independently trained seeds.}",
            "\\label{tab:matched-auroc}",
            "\\begin{tabular}{lcc}",
            "\\toprule",
            "Score & AIDER & Hurricane Damage \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    (TABLES / "matched_auroc.tex").write_text(text)


def write_inflation_table(data: dict) -> None:
    rows = []
    for seed in data["protocol"]["seeds"]:
        cells = []
        for corpus in ("aider", "hurricane"):
            value = data[corpus]["seeds"][str(seed)]["comparisons"][
                "privileged_minus_received"
            ]
            cells.append(
                f"{value['mean']:.3f} "
                f"[{value['ci95'][0]:.3f}, {value['ci95'][1]:.3f}]"
            )
        rows.append(f"{seed} & {cells[0]} & {cells[1]} \\\\")
    text = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Paired oracle--received gap in error AUROC. "
            "Brackets give 95\\% paired bootstrap intervals from 10,000 "
            "replicates.}",
            "\\label{tab:inflation}",
            "\\begin{tabular}{lcc}",
            "\\toprule",
            "Seed & AIDER & Hurricane Damage \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    (TABLES / "privileged_inflation.tex").write_text(text)


def write_anchor_matched_table(data: dict) -> None:
    rows = []
    for seed in data["protocol"]["seeds"]:
        values = []
        for corpus in ("aider", "hurricane"):
            comparison = data[corpus]["seeds"][str(seed)]["comparisons"][
                "fine_reference_minus_received"
            ]
            values.append(
                f"{comparison['mean']:.3f} "
                f"[{comparison['ci95'][0]:.3f}, {comparison['ci95'][1]:.3f}]"
            )
        rows.append(f"{seed} & {values[0]} & {values[1]} \\\\")
    text = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Post-hoc anchor-matched mechanism sensitivity. Both "
            "scores anchor on the received $s=8$ prediction and use three "
            "comparisons; the table reports fine-reference minus further-"
            "degradation error AUROC with 95\\% paired bootstrap intervals.}",
            "\\label{tab:anchor-matched}",
            "\\begin{tabular}{lcc}",
            "\\toprule",
            "Seed & AIDER & Hurricane Damage \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    (TABLES / "anchor_matched_sensitivity.tex").write_text(text)


def write_leakage_table() -> None:
    aider = json.loads(AIDER_AUDIT.read_text())
    hurricane = json.loads(HURRICANE_AUDIT.read_text())
    rows = [
        (
            "AIDER",
            len(aider["conflicting_groups"]),
            len(aider["excluded_paths"]),
            sum(aider["corrected_counts"].values()),
        ),
        (
            "Hurricane Damage",
            len(hurricane["cross_split_groups"]),
            len(hurricane["excluded_indexes"]),
            sum(hurricane["corrected_counts"].values()),
        ),
    ]
    body = [
        f"{name} & {groups} & {excluded} & {retained} \\\\"
        for name, groups, excluded, retained in rows
    ]
    text = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Encoded-byte cross-split duplicate audit. Every member "
            "of each listed group was excluded before retraining.}",
            "\\label{tab:hash-audit}",
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "Dataset & Groups & Excluded & Retained \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    (TABLES / "hash_audit.tex").write_text(text)


def make_figure(data: dict) -> None:
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    x = np.arange(len(SCORES))
    colours = ["#4C78A8"] * (len(SCORES) - 1) + ["#E45756"]
    for axis, corpus, title in zip(
        axes[:2],
        ("aider", "hurricane"),
        ("(a) AIDER", "(b) Hurricane Damage"),
    ):
        means = [
            data[corpus]["aggregate"]["scores"][key]["mean_auroc"]
            for key, _ in SCORES
        ]
        standard_deviations = [
            data[corpus]["aggregate"]["scores"][key]["std_auroc"]
            for key, _ in SCORES
        ]
        axis.bar(
            x,
            means,
            yerr=standard_deviations,
            color=colours,
            capsize=2,
            linewidth=0,
        )
        axis.set_ylim(0.35, 1.03)
        axis.set_xticks(x, FIGURE_LABELS, rotation=45, ha="right")
        axis.set_ylabel("Error AUROC")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    seeds = data["protocol"]["seeds"]
    width = 0.34
    axes[2].bar(
        np.arange(len(seeds)) - width / 2,
        data["aider"]["aggregate"]["privileged_inflation"]["values"],
        width,
        label="AIDER",
        color="#4C78A8",
    )
    axes[2].bar(
        np.arange(len(seeds)) + width / 2,
        data["hurricane"]["aggregate"]["privileged_inflation"]["values"],
        width,
        label="Hurricane",
        color="#F2CF5B",
    )
    axes[2].axhline(0, color="black", linewidth=0.7)
    axes[2].set_xticks(np.arange(len(seeds)), [str(seed) for seed in seeds])
    axes[2].set_xlabel("Training seed")
    axes[2].set_ylabel("Reference $-$ received AUROC")
    axes[2].set_title("(c) Oracle--received gap")
    axes[2].legend(frameon=False, fontsize=7)
    axes[2].grid(axis="y", alpha=0.25)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(
        FIGURES / "information_audit.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def write_measured_table(training: dict, paired: dict) -> None:
    site_rows = []
    sites = {site["id"]: site for site in paired["protocol"]["sites"]}
    for site_id, lift in paired["aggregate"][
        "site_mean_satellite_auroc_lifts"
    ].items():
        site = sites[site_id]
        site_rows.append(
            f"{site_id.replace('-', ' ').title()} & {site['event']} & "
            f"{paired['pair_counts'][site_id]['n']} & "
            f"{100 * site['uas_gsd_m']:.2f} & "
            f"{100 * site['satellite_gsd_m']:.2f} & {lift:.3f} \\\\"
        )
    site_rows.append(
        "\\midrule\nPooled & Two events & "
        f"{sum(row['n'] for row in paired['pair_counts'].values())} & "
        "-- & -- & "
        f"{paired['aggregate']['mean_satellite_auroc_lift']:.3f} \\\\"
    )
    text = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Held-out same-building measured-GSD evaluation. GSD is "
            "reported in cm\\,px$^{-1}$. The final column is the mean "
            "received-consistency minus confidence error-AUROC difference "
            "across three seeds.}",
            "\\label{tab:measured-gsd}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{llrrrr}",
            "\\toprule",
            "Acquisition & Event & Pairs & UAS GSD & Satellite GSD & AUROC lift \\\\",
            "\\midrule",
            *site_rows,
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{table}",
            "",
        ]
    )
    (TABLES / "measured_gsd.tex").write_text(text)

    internal = training["aggregate"]
    pooled = paired["seeds"]
    values = {
        "UAS accuracy": [
            pooled[str(seed)]["pooled"]["uas"]["accuracy"]
            for seed in paired["protocol"]["seeds"]
        ],
        "Satellite accuracy": [
            pooled[str(seed)]["pooled"]["satellite"]["accuracy"]
            for seed in paired["protocol"]["seeds"]
        ],
        "Satellite confidence AUROC": [
            pooled[str(seed)]["pooled"]["satellite"]["auroc_confidence"]
            for seed in paired["protocol"]["seeds"]
        ],
        "Satellite received-consistency AUROC": [
            pooled[str(seed)]["pooled"]["satellite"]["auroc_rcg"]
            for seed in paired["protocol"]["seeds"]
        ],
    }
    rows = [
        (
            f"Guarded internal accuracy & {internal['accuracy']['mean']:.3f} "
            f"$\\pm$ {internal['accuracy']['std']:.3f} \\\\"
        ),
        (
            "Guarded internal majority accuracy & "
            f"{internal['majority_accuracy']['mean']:.3f} \\\\"
        ),
    ]
    rows.extend(
        f"{name} & {np.mean(metric):.3f} $\\pm$ "
        f"{np.std(metric, ddof=1):.3f} \\\\"
        for name, metric in values.items()
    )
    text = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Intersection-controlled CRASAR classifier and pooled held-out "
            "paired results. Values are mean $\\pm$ sample SD across seeds.}",
            "\\label{tab:crasar-summary}",
            "\\begin{tabular}{lc}",
            "\\toprule",
            "Endpoint & Value \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    (TABLES / "crasar_summary.tex").write_text(text)


def make_measured_figure(paired: dict) -> None:
    seeds = [str(seed) for seed in paired["protocol"]["seeds"]]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25))
    modalities = ("uas", "satellite")
    accuracy = [
        [
            paired["seeds"][seed]["pooled"][modality]["accuracy"]
            for seed in seeds
        ]
        for modality in modalities
    ]
    axes[0].bar(
        [0, 1],
        [np.mean(row) for row in accuracy],
        yerr=[np.std(row, ddof=1) for row in accuracy],
        color=["#4C78A8", "#F2CF5B"],
        capsize=3,
    )
    axes[0].axhline(
        paired["aggregate"]["pooled_uas_majority_accuracy"][0],
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="Pooled majority",
    )
    axes[0].set_xticks([0, 1], ["UAS", "Satellite"])
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.3, 0.85)
    axes[0].set_title("Paired classification")
    axes[0].legend(frameon=False, fontsize=7)
    confidence = [
        paired["seeds"][seed]["pooled"]["satellite"]["auroc_confidence"]
        for seed in seeds
    ]
    rcg = [
        paired["seeds"][seed]["pooled"]["satellite"]["auroc_rcg"]
        for seed in seeds
    ]
    axes[1].bar(
        [0, 1],
        [np.mean(confidence), np.mean(rcg)],
        yerr=[np.std(confidence, ddof=1), np.std(rcg, ddof=1)],
        color=["#4C78A8", "#E45756"],
        capsize=3,
    )
    axes[1].set_xticks([0, 1], ["Confidence", "Received\nconsistency"])
    axes[1].set_ylabel("Satellite error AUROC")
    axes[1].set_ylim(0.3, 0.75)
    axes[1].set_title("Pooled error ranking")
    site_lifts = paired["aggregate"]["site_mean_satellite_auroc_lifts"]
    short_labels = ["Harlem", "McGregor", "Mexico 13", "Mexico 14"]
    axes[2].bar(
        np.arange(4),
        list(site_lifts.values()),
        color=[
            "#4C78A8" if value >= 0 else "#E45756"
            for value in site_lifts.values()
        ],
    )
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_xticks(
        np.arange(4), short_labels, rotation=35, ha="right"
    )
    axes[2].set_ylabel("RCG $-$ confidence AUROC")
    axes[2].set_title("Acquisition heterogeneity")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout(w_pad=1.1)
    fig.savefig(
        FIGURES / "measured_gsd_audit.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def write_architecture_operator_table(mobile: dict, resnet: dict) -> None:
    labels = {
        "bicubic": "Bicubic",
        "bilinear": "Bilinear",
        "nearest": "Nearest",
        "box": "Box",
    }
    rows = []
    for operator, label in labels.items():
        mobile_row = mobile["aggregate"]["operators"][operator]
        aider_row = resnet["aggregate"]["aider"][operator]
        hurricane_row = resnet["aggregate"]["hurricane"][operator]
        rows.append(
            f"{label} & {mobile_row['mean']:.3f} $\\pm$ "
            f"{mobile_row['std']:.3f} & {aider_row['mean']:.3f} $\\pm$ "
            f"{aider_row['std']:.3f} & {hurricane_row['mean']:.3f} $\\pm$ "
            f"{hurricane_row['std']:.3f} \\\\"
        )
    text = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Prospective architecture/operator replication of the "
            "anchor-matched fine-reference minus received-image error-AUROC "
            "gap. Values are mean $\\pm$ sample SD over three independently "
            "trained seeds.}",
            "\\label{tab:architecture-operator}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Operator & MobileNet AIDER & ResNet AIDER & ResNet Hurricane \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{table}",
            "",
        ]
    )
    (TABLES / "architecture_operator_replication.tex").write_text(text)


def make_architecture_operator_figure(mobile: dict, resnet: dict) -> None:
    operators = ["bicubic", "bilinear", "nearest", "box"]
    labels = ["Bicubic", "Bilinear", "Nearest", "Box"]
    series = [
        (
            "MobileNet AIDER",
            [mobile["aggregate"]["operators"][op]["mean"] for op in operators],
            [mobile["aggregate"]["operators"][op]["std"] for op in operators],
            "#4C78A8",
        ),
        (
            "ResNet AIDER",
            [resnet["aggregate"]["aider"][op]["mean"] for op in operators],
            [resnet["aggregate"]["aider"][op]["std"] for op in operators],
            "#F2CF5B",
        ),
        (
            "ResNet Hurricane",
            [
                resnet["aggregate"]["hurricane"][op]["mean"]
                for op in operators
            ],
            [
                resnet["aggregate"]["hurricane"][op]["std"]
                for op in operators
            ],
            "#E45756",
        ),
    ]
    x = np.arange(len(operators))
    width = 0.25
    fig, axis = plt.subplots(figsize=(7.0, 2.65))
    for index, (name, means, standard_deviations, colour) in enumerate(series):
        axis.bar(
            x + (index - 1) * width,
            means,
            width,
            yerr=standard_deviations,
            capsize=2,
            label=name,
            color=colour,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Fine-reference $-$ received AUROC")
    axis.set_title("Architecture and degradation-operator replication")
    axis.legend(frameon=False, ncol=3, fontsize=7)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "architecture_operator_replication.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def write_idalia_table(data: dict) -> None:
    rows = []
    for seed in data["protocol"]["seeds"]:
        result = data["results"][str(seed)]
        interval = result["cluster_bootstrap"]["ci95"]
        rows.append(
            f"{seed} & {result['uas']['balanced_accuracy']:.3f} & "
            f"{result['crewed']['balanced_accuracy']:.3f} & "
            f"{result['crewed_rcg_lift']:.3f} & "
            f"[{interval[0]:.3f}, {interval[1]:.3f}] \\\\"
        )
    text = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Prospective third-event Hurricane Idalia sensitivity "
            "on 458 same-building UAS/crewed-aircraft pairs. The final columns "
            "report received-consistency minus confidence error-AUROC and its "
            "joint-overlap-cluster bootstrap interval.}",
            "\\label{tab:idalia}",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Seed & UAS balanced acc. & Crewed balanced acc. & AUROC lift & 95\\% CI \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    (TABLES / "idalia_sensitivity.tex").write_text(text)


def write_reveal_mask_table(data: dict) -> None:
    rows = []
    labels = {
        "bicubic": "Bicubic",
        "bilinear": "Bilinear",
        "nearest": "Nearest",
        "box": "Box",
    }
    for operator, label in labels.items():
        values = {}
        probabilities = []
        for corpus in ("aider", "hurricane"):
            gaps = [
                data["results"][corpus][str(seed)][operator][
                    "aligned_minus_masked_mean"
                ]
                for seed in (101, 202, 303)
            ]
            values[corpus] = (np.mean(gaps), np.std(gaps, ddof=1))
            probabilities.extend(
                data["results"][corpus][str(seed)][operator][
                    "empirical_probability"
                ]
                for seed in (101, 202, 303)
            )
        rows.append(
            f"{label} & {values['aider'][0]:.3f} $\\pm$ "
            f"{values['aider'][1]:.3f} & {values['hurricane'][0]:.3f} "
            f"$\\pm$ {values['hurricane'][1]:.3f} & "
            f"{max(probabilities):.4f} \\\\"
        )
    text = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Pre-specified fine-reference reveal/mask experiment. "
            "Values are aligned-minus-masked-mean error AUROC across three "
            "ResNet seeds; the final column is the maximum empirical one-sided "
            "probability across both corpora and all seeds.}",
            "\\label{tab:reveal-mask}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Operator & AIDER gap & Hurricane gap & Max. probability \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    (TABLES / "reference_reveal_mask.tex").write_text(text)


def main() -> None:
    data = json.loads(SOURCE.read_text())
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_score_table(data)
    write_inflation_table(data)
    write_anchor_matched_table(data)
    write_leakage_table()
    make_figure(data)
    if CRASAR_SOURCE.exists() and PAIRED_SOURCE.exists():
        training = json.loads(CRASAR_SOURCE.read_text())
        paired = json.loads(PAIRED_SOURCE.read_text())
        write_measured_table(training, paired)
        make_measured_figure(paired)
    if MOBILENET_SOURCE.exists() and RESNET_OPERATOR_SOURCE.exists():
        mobile = json.loads(MOBILENET_SOURCE.read_text())
        resnet = json.loads(RESNET_OPERATOR_SOURCE.read_text())
        write_architecture_operator_table(mobile, resnet)
        make_architecture_operator_figure(mobile, resnet)
    if IDALIA_SOURCE.exists():
        write_idalia_table(json.loads(IDALIA_SOURCE.read_text()))
    if REVEAL_MASK_SOURCE.exists():
        write_reveal_mask_table(json.loads(REVEAL_MASK_SOURCE.read_text()))
    print("AUDIT_ARTIFACTS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
