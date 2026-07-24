#!/usr/bin/env python3
"""Register greatness-strengthening claims and artifacts in JSONL ledgers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "experiments/derived/greatness_strengthening"
CLAIMS = ROOT / "evidence/claim-ledger.jsonl"
ARTIFACTS = ROOT / "evidence/artifact-ledger.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path: str, selector: str | None = None) -> dict:
    value = {"path": path, "sha256": sha256(ROOT / path)}
    if selector:
        value["selector"] = selector
    return value


def upsert_jsonl(path: Path, records: list[dict], key: str) -> None:
    current = {}
    order = []
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            current[record[key]] = record
            order.append(record[key])
    for record in records:
        if record[key] not in current:
            order.append(record[key])
        current[record[key]] = record
    path.write_text(
        "\n".join(json.dumps(current[item]) for item in order) + "\n"
    )


def main() -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    primary_path = "experiments/derived/greatness_strengthening/primary_multiseed.json"
    measured_path = (
        "experiments/derived/greatness_strengthening/"
        "crasar/measured_gsd_crasar.json"
    )
    paired_path = (
        "experiments/derived/greatness_strengthening/"
        "crasar/paired_measured_gsd.json"
    )
    multisite_path = (
        "experiments/derived/greatness_strengthening/"
        "crasar/multisite_paired_gsd.json"
    )
    summary_path = (
        "experiments/derived/greatness_strengthening/strengthening_summary.json"
    )
    primary = json.loads((ROOT / primary_path).read_text())
    measured = json.loads((ROOT / measured_path).read_text())
    paired = json.loads((ROOT / paired_path).read_text())
    multisite = json.loads((ROOT / multisite_path).read_text())
    summary = json.loads((ROOT / summary_path).read_text())

    claims = [
        {
            "claim_id": "C020",
            "claim": (
                "Across AIDER ResNet-18 seeds 101/202/303, s=8 RCG error "
                "AUROC exceeds confidence for all seeds; mean lift "
                f"{primary['aider']['aggregate']['lift_rcg_minus_confidence']['mean']:.3f} "
                f"(SD {primary['aider']['aggregate']['lift_rcg_minus_confidence']['std']:.3f})."
            ),
            "type": "quantitative",
            "scope": "AIDER frozen test n=970, ImageNet-pretrained ResNet-18, 3 seeds",
            "status": "verified",
            "verified_at": now,
            "source_artifacts": [source(primary_path, "aider")],
            "analysis_command": (
                "python experiments/code/run_primary_multiseed.py "
                "--datasets aider --eval-only"
            ),
            "result": primary["aider"]["aggregate"],
            "paper_locations": ["abstract", "results", "discussion"],
        },
        {
            "claim_id": "C021",
            "claim": (
                "Across Hurricane ResNet-18 seeds 101/202/303, s=8 RCG error "
                "AUROC exceeds confidence for all seeds; mean lift "
                f"{primary['hurricane']['aggregate']['lift_rcg_minus_confidence']['mean']:.3f} "
                f"(SD {primary['hurricane']['aggregate']['lift_rcg_minus_confidence']['std']:.3f})."
            ),
            "type": "quantitative",
            "scope": "Hurricane frozen test n=2000, ImageNet-pretrained ResNet-18, 3 seeds",
            "status": "verified",
            "verified_at": now,
            "source_artifacts": [source(primary_path, "hurricane")],
            "analysis_command": (
                "python experiments/code/run_primary_multiseed.py "
                "--datasets hurricane --eval-only"
            ),
            "result": primary["hurricane"]["aggregate"],
            "paper_locations": ["abstract", "results", "discussion"],
        },
        {
            "claim_id": "C022",
            "claim": (
                "RCG exceeds the EO-specific kNN feature-distance OOD baseline "
                "for every AIDER and Hurricane ResNet seed at s=8; mean lifts "
                f"are {primary['aider']['aggregate']['lift_rcg_minus_knn']['mean']:.3f} "
                "and "
                f"{primary['hurricane']['aggregate']['lift_rcg_minus_knn']['mean']:.3f}."
            ),
            "type": "quantitative",
            "scope": "AIDER and Hurricane identical models/splits/seeds, k=10 native train bank",
            "status": "verified",
            "verified_at": now,
            "source_artifacts": [source(primary_path)],
            "analysis_command": (
                "python experiments/code/run_primary_multiseed.py "
                "--datasets aider hurricane --eval-only"
            ),
            "result": {
                "aider": primary["aider"]["aggregate"]["lift_rcg_minus_knn"],
                "hurricane": primary["hurricane"]["aggregate"][
                    "lift_rcg_minus_knn"
                ],
            },
            "paper_locations": ["related-work", "results", "discussion"],
        },
        {
            "claim_id": "C023",
            "claim": (
                "The initial unpaired three-orthomosaic measured-GSD audit did "
                "not show monotonic accuracy decline; its preregistered W6 "
                "criterion failed and motivated a paired course correction."
            ),
            "type": "quantitative",
            "scope": "CRASAR UAS three-orthomosaic spatial-block holdout, 3 seeds",
            "status": "verified",
            "verified_at": now,
            "source_artifacts": [source(measured_path)],
            "analysis_command": (
                "python experiments/code/run_measured_gsd_crasar.py --epochs 5"
            ),
            "result": measured["aggregate"],
            "paper_locations": ["results", "limitations", "appendix"],
        },
        {
            "claim_id": "C024",
            "claim": (
                "On 67 held-out same-building CRASAR pairs, measured GSD "
                "increases from 4.672 to 30.518 cm/px; mean accuracy falls by "
                f"{paired['aggregate']['paired_accuracy_drop']:.3f}, while "
                "satellite-view RCG error AUROC exceeds confidence by "
                f"{paired['aggregate']['satellite_rcg_lift']:.3f} across 3 seeds."
            ),
            "type": "quantitative",
            "scope": "CRASAR Harlem Heights paired UAS/satellite same-label buildings, 3 seeds",
            "status": "verified",
            "verified_at": now,
            "source_artifacts": [source(paired_path)],
            "analysis_command": (
                "python experiments/code/run_paired_measured_gsd.py"
            ),
            "result": paired["aggregate"],
            "paper_locations": ["abstract", "results", "discussion"],
        },
        {
            "claim_id": "C025",
            "claim": (
                "Across 125 held-out same-building CRASAR pairs at two sites, "
                "UAS GSD 3.839--4.672 cm/px versus satellite GSD 30.518 "
                "cm/px reduces pooled mean accuracy by "
                f"{multisite['aggregate']['pooled']['accuracy_drop']:.3f}; "
                "satellite-view RCG exceeds confidence AUROC by "
                f"{multisite['aggregate']['pooled']['satellite_rcg_lift']:.3f} "
                "across three seeds."
            ),
            "type": "quantitative",
            "scope": "CRASAR two-site pooled paired UAS/satellite same-label buildings, 3 seeds",
            "status": "verified",
            "verified_at": now,
            "source_artifacts": [source(multisite_path)],
            "analysis_command": (
                "python experiments/code/run_multisite_paired_gsd.py"
            ),
            "result": multisite["aggregate"],
            "paper_locations": ["abstract", "results", "discussion"],
        },
    ]
    upsert_jsonl(CLAIMS, claims, "claim_id")

    checkpoint_sources = []
    for dataset in ("aider", "hurricane"):
        for seed in (101, 202, 303):
            relative = (
                "experiments/derived/greatness_strengthening/"
                f"{dataset}/seed_{seed}/best.pt"
            )
            checkpoint_sources.append(source(relative))
    for seed in (101, 202, 303):
        relative = (
            "experiments/derived/greatness_strengthening/"
            f"crasar/seed_{seed}/best.pt"
        )
        checkpoint_sources.append(source(relative))

    artifact_specs = [
        (
            "A-strength-primary",
            "algorithm-output",
            primary_path,
            "experiments/code/run_primary_multiseed.py",
            ["C020", "C021", "C022"],
            checkpoint_sources[:6],
        ),
        (
            "A-strength-measured-unpaired",
            "algorithm-output",
            measured_path,
            "experiments/code/run_measured_gsd_crasar.py",
            ["C023"],
            checkpoint_sources[6:],
        ),
        (
            "A-strength-measured-paired",
            "algorithm-output",
            paired_path,
            "experiments/code/run_paired_measured_gsd.py",
            ["C024"],
            checkpoint_sources[6:],
        ),
        (
            "A-strength-summary",
            "algorithm-output",
            summary_path,
            "experiments/code/generate_strengthening_artifacts.py",
            ["C020", "C021", "C022", "C023", "C024", "C025"],
            [
                source(primary_path),
                source(measured_path),
                source(paired_path),
                source(multisite_path),
            ],
        ),
        (
            "A-tab-primary-multiseed",
            "table",
            "paper/tables/primary_multiseed.tex",
            "experiments/code/generate_strengthening_artifacts.py",
            ["C020", "C021", "C022"],
            [source(primary_path)],
        ),
        (
            "A-tab-measured-gsd",
            "table",
            "paper/tables/measured_gsd.tex",
            "experiments/code/generate_strengthening_artifacts.py",
            ["C025"],
            [source(multisite_path)],
        ),
        (
            "A-tab-measured-gsd-unpaired",
            "table",
            "paper/tables/measured_gsd_unpaired.tex",
            "experiments/code/generate_strengthening_artifacts.py",
            ["C023"],
            [source(measured_path)],
        ),
        (
            "A-fig-greatness-strengthening",
            "figure",
            "paper/figures/greatness_strengthening.pdf",
            "experiments/code/generate_strengthening_artifacts.py",
            ["C020", "C021", "C022", "C025"],
            [source(primary_path), source(multisite_path)],
        ),
        (
            "A-strength-measured-multisite",
            "algorithm-output",
            multisite_path,
            "experiments/code/run_multisite_paired_gsd.py",
            ["C025"],
            checkpoint_sources[6:],
        ),
    ]
    artifacts = []
    for (
        artifact_id,
        artifact_type,
        path,
        generator,
        claim_ids,
        sources,
    ) in artifact_specs:
        artifacts.append(
            {
                "artifact_id": artifact_id,
                "type": artifact_type,
                "path": path,
                "sha256": sha256(ROOT / path),
                "generator_code": generator,
                "generator_code_sha256": sha256(ROOT / generator),
                "generator_command": f"python {generator}",
                "source_artifacts": sources,
                "claim_ids": claim_ids,
                "justification": "Greatness-strengthening evidence",
                "latex_reference": "strengthening",
                "verified_at": now,
            }
        )
    upsert_jsonl(ARTIFACTS, artifacts, "artifact_id")
    print(f"registered {len(claims)} claims and {len(artifacts)} artifacts")


if __name__ == "__main__":
    main()
