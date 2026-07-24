"""Generate the two numerical tables corrected during peer-review revision."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def fmt(value: float) -> str:
    return f"{value:.3f}"


def scale_ablation() -> str:
    with (ROOT / "experiments/derived/aider_rcg/confirmatory_summary.json").open() as f:
        scales = json.load(f)["local_ckpt"]["scales"]

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Operating $s$ & Acc & Err-conf & AUROC$_{1-c}$ & AUROC$_{JS}$ / Lift \\",
        r"\midrule",
    ]
    for scale in ("1", "2", "4", "8"):
        row = scales[scale]
        lift = row["lift_js_minus_1mconf"]
        lines.append(
            f"{scale} & {fmt(row['acc'])} & {fmt(row['err_conf'])} & "
            f"{fmt(row['auroc_1mconf'])} & {fmt(row['auroc_js'])} / "
            f"${lift:+.3f}$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def threshold_transfer() -> str:
    with (ROOT / "experiments/derived/aider_rcg/threshold_transfer.json").open() as f:
        aider = json.load(f)
    with (ROOT / "experiments/derived/strengthen_unambiguous/summary.json").open() as f:
        hurricane = json.load(f)["hurricane"]["transfer"]

    a1, a8 = aider["transfer_to_test_s1"], aider["transfer_to_test_s8"]
    h1, h8 = hurricane["s1"], hurricane["s8"]
    rows = [
        (
            "AIDER MobileNet",
            "Confidence",
            a1["conf"]["coverage"],
            a1["conf"]["fcr"],
            a8["conf"]["coverage"],
            a8["conf"]["fcr"],
        ),
        (
            "AIDER MobileNet",
            "RCG-JS",
            a1["js_rcg"]["coverage"],
            a1["js_rcg"]["fcr"],
            a8["js_rcg"]["coverage"],
            a8["js_rcg"]["fcr"],
        ),
        (
            "Hurricane ResNet",
            "Confidence",
            h1["conf"]["coverage"],
            h1["conf"]["fcr"],
            h8["conf"]["coverage"],
            None,
        ),
        (
            "Hurricane ResNet",
            "RCG-JS",
            h1["js"]["coverage"],
            h1["js"]["fcr"],
            h8["js"]["coverage"],
            h8["js"]["fcr"],
        ),
    ]

    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Corpus & Gate & Cov.\ $s{=}1$ & FCR $s{=}1$ & Cov.\ $s{=}8$ & FCR $s{=}8$ \\",
        r"\midrule",
    ]
    for corpus, gate, cov1, fcr1, cov8, fcr8 in rows:
        fcr8_text = "---" if fcr8 is None else fmt(fcr8)
        lines.append(
            f"{corpus} & {gate} & {fmt(cov1)} & {fmt(fcr1)} & "
            f"{fmt(cov8)} & {fcr8_text} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    outputs = {
        "scale_ablation.tex": scale_ablation(),
        "threshold_transfer.tex": threshold_transfer(),
    }
    paper_dirs = [ROOT / "paper"]
    if (ROOT / "paper-ijrs").exists():
        paper_dirs.append(ROOT / "paper-ijrs")
    for paper_dir in paper_dirs:
        table_dir = paper_dir / "tables"
        for name, content in outputs.items():
            path = table_dir / name
            path.write_text(content, encoding="utf-8")
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
