#!/usr/bin/env python3
"""Register execution, claim, and publication-artifact evidence."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TERMINALS = Path(
    "/Users/akshay.sharma/.cursor/projects/"
    "Users-akshay-sharma-Projects-paper-activities/terminals"
)
IMPORTED = Path("experiments/imported")
NOW = "2026-07-24T22:30:00Z"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: str) -> dict:
    full = ROOT / path
    return {"path": path, "sha256": sha256(full)}


def terminal_times(terminal_id: str) -> tuple[str, str, int]:
    text = (TERMINALS / f"{terminal_id}.txt").read_text()
    started = re.search(r"^started_at: (.+)$", text, re.MULTILINE)
    ended = re.search(r"^ended_at: (.+)$", text, re.MULTILINE)
    exits = re.findall(r"^exit_code: (\d+)$", text, re.MULTILINE)
    if not started or not ended or not exits:
        raise ValueError(f"incomplete terminal metadata: {terminal_id}")
    return started.group(1), ended.group(1), int(exits[-1])


def local_times(log_path: str, seconds: float) -> tuple[str, str]:
    completed = datetime.fromtimestamp(
        (ROOT / log_path).stat().st_mtime, tz=timezone.utc
    )
    started = completed - timedelta(seconds=seconds)
    return started.isoformat(), completed.isoformat()


def run_from_terminal(
    run_id: str,
    node_id: str | None,
    execution_kind: str,
    command: str,
    terminal_id: str,
    log_path: str,
    outputs: list[str],
    status: str = "succeeded",
    cwd: str = "../disaster-rcg-gsd",
) -> dict:
    started, completed, exit_code = terminal_times(terminal_id)
    return {
        "run_id": run_id,
        "node_id": node_id,
        "execution_kind": execution_kind,
        "command": command,
        "cwd": cwd,
        "started_at": started,
        "completed_at": completed,
        "status": status,
        "exit_code": exit_code,
        "log_path": log_path,
        "log_sha256": sha256(ROOT / log_path),
        "output_artifacts": [record(path) for path in outputs],
    }


def local_run(
    run_id: str,
    execution_kind: str,
    command: str,
    log_path: str,
    outputs: list[str],
    seconds: float,
    cwd: str = ".",
) -> dict:
    started, completed = local_times(log_path, seconds)
    return {
        "run_id": run_id,
        "node_id": None,
        "execution_kind": execution_kind,
        "command": command,
        "cwd": cwd,
        "started_at": started,
        "completed_at": completed,
        "status": "succeeded",
        "exit_code": 0,
        "log_path": log_path,
        "log_sha256": sha256(ROOT / log_path),
        "output_artifacts": [record(path) for path in outputs],
    }


def imported(relative: str) -> str:
    return (IMPORTED / relative).as_posix()


def source(path: str) -> dict:
    return record(path)


def main() -> None:
    aider_outputs = [
        imported(
            f"experiments/derived/greatness_iteration3/aider/seed_{seed}/best.pt"
        )
        for seed in (101, 202, 303)
    ]
    hurricane_outputs = [
        imported(
            "experiments/derived/greatness_iteration3/"
            f"hurricane/seed_{seed}/best.pt"
        )
        for seed in (101, 202, 303)
    ]
    crasar_outputs = [
        imported(
            "experiments/derived/greatness_iteration3/"
            f"crasar/seed_{seed}/best.pt"
        )
        for seed in (101, 202, 303)
    ] + [
        imported(
            "experiments/derived/greatness_iteration3/"
            "crasar/leakage_free_crasar.json"
        )
    ]
    paired_outputs = [
        imported(
            "experiments/derived/greatness_iteration3/"
            "crasar/leakage_free_paired_gsd.json"
        )
    ] + [
        imported(
            "experiments/derived/greatness_iteration3/crasar/"
            f"paired_evaluation/{site}/pair_manifest.json"
        )
        for site in (
            "harlem-heights",
            "mcgregor-college-parkway-south-1",
            "mexico-beach-2018-10-13",
            "mexico-beach-2018-10-14",
        )
    ]
    information_score_outputs = [
        imported(
            "experiments/derived/greatness_iteration3/information_scores/"
            f"{corpus}_seed_{seed}.npz"
        )
        for corpus in ("aider", "hurricane")
        for seed in (101, 202, 303)
    ]
    paired_v2_outputs = [
        imported(
            "experiments/derived/greatness_iteration3/crasar/"
            "leakage_free_paired_gsd_v2.json"
        ),
        imported(
            "experiments/derived/greatness_iteration3/crasar/"
            "paired_scores/paired_score_index.json"
        ),
    ] + [
        imported(
            "experiments/derived/greatness_iteration3/crasar/"
            f"paired_scores/seed_{seed}.npz"
        )
        for seed in (101, 202, 303)
    ]
    mobile_outputs = [
        "experiments/derived/architecture_replication/"
        "mobilenet_aider_operator_audit.json"
    ] + [
        f"experiments/derived/architecture_replication/seed_{seed}/{name}"
        for seed in (101, 202, 303)
        for name in (
            "best.pt",
            "history.json",
            "scores/bicubic.npz",
            "scores/bilinear.npz",
            "scores/nearest.npz",
            "scores/box.npz",
        )
    ]
    resnet_operator_outputs = [
        "experiments/derived/architecture_replication/"
        "resnet_operator_sensitivity.json"
    ] + [
        "experiments/derived/architecture_replication/resnet_scores/"
        f"{corpus}/seed_{seed}/{operator}.npz"
        for corpus in ("aider", "hurricane")
        for seed in (101, 202, 303)
        for operator in ("bicubic", "bilinear", "nearest", "box")
    ]
    reveal_mask_outputs = [
        "experiments/derived/reference_reveal_mask/reveal_mask_summary.json"
    ] + [
        "experiments/derived/reference_reveal_mask/"
        f"{corpus}/seed_{seed}/{operator}.npz"
        for corpus in ("aider", "hurricane")
        for seed in (101, 202, 303)
        for operator in ("bicubic", "bilinear", "nearest", "box")
    ]
    idalia_outputs = [
        "experiments/derived/idalia_paired/idalia_paired_sensitivity.json",
        "experiments/derived/idalia_paired/selection_audit.json",
        "experiments/derived/idalia_paired/score_index.json",
    ] + [
        f"experiments/derived/idalia_paired/seed_{seed}.npz"
        for seed in (101, 202, 303)
    ]
    runs = [
        local_run(
            "R-aider-dedup-prep",
            "data-preparation",
            "python experiments/code/prepare_aider_dedup.py",
            imported("experiments/logs/R-iter3-aider-dedup-prep.log"),
            [
                imported(
                    "experiments/derived/greatness_iteration3/"
                    "aider_splits_dedup.json"
                ),
                imported(
                    "experiments/derived/greatness_iteration3/"
                    "aider_dedup_audit.json"
                ),
            ],
            2.0,
            cwd="../disaster-rcg-gsd",
        ),
        local_run(
            "R-hurricane-dedup-prep",
            "data-preparation",
            "python experiments/code/prepare_hurricane_dedup.py",
            imported("experiments/logs/R-iter3-hurricane-dedup-prep.log"),
            [
                imported(
                    "experiments/derived/greatness_iteration3/"
                    "hurricane_splits_dedup.json"
                ),
                imported(
                    "experiments/derived/greatness_iteration3/"
                    "hurricane_dedup_audit.json"
                ),
            ],
            5.0,
            cwd="../disaster-rcg-gsd",
        ),
        run_from_terminal(
            "R-aider-dedup-train",
            "n002",
            "training",
            "python experiments/code/run_primary_multiseed.py --datasets aider "
            "--aider-splits experiments/derived/greatness_iteration3/"
            "aider_splits_dedup.json",
            "489477",
            imported("experiments/logs/R-iter3-train-aider.log"),
            aider_outputs,
        ),
        run_from_terminal(
            "R-hurricane-dedup-train",
            "n002",
            "training",
            "python experiments/code/run_primary_multiseed.py --datasets "
            "hurricane --hurricane-splits experiments/derived/"
            "greatness_iteration3/hurricane_splits_dedup.json",
            "489478",
            imported("experiments/logs/R-iter3-train-hurricane.log"),
            hurricane_outputs,
        ),
        run_from_terminal(
            "R-information-audit-v1-failed",
            "n003",
            "evaluation",
            "python experiments/code/run_information_matched_audit.py",
            "489480",
            imported("experiments/logs/R-iter3-information-audit.log"),
            [],
            status="failed",
        ),
        run_from_terminal(
            "R-information-audit-v3",
            "n003",
            "evaluation",
            "python experiments/code/run_information_matched_audit.py",
            "489485",
            imported("experiments/logs/R-iter3-information-audit-v3.log"),
            [
                imported(
                    "experiments/derived/greatness_iteration3/"
                    "information_matched_audit.json"
                )
            ]
            + information_score_outputs,
        ),
        run_from_terminal(
            "R-crasar-prepare-v1-failed",
            "n004",
            "data-preparation",
            "python experiments/code/run_leakage_free_crasar.py --prepare-only",
            "489479",
            imported("experiments/logs/R-iter3-crasar-prepare.log"),
            [],
            status="failed",
        ),
        run_from_terminal(
            "R-crasar-train",
            "n004",
            "training",
            "python experiments/code/run_leakage_free_crasar.py --epochs 12",
            "489482",
            imported("experiments/logs/R-iter3-train-crasar.log"),
            crasar_outputs,
        ),
        run_from_terminal(
            "R-paired-gsd",
            "n005",
            "evaluation",
            "python experiments/code/run_leakage_free_paired_gsd.py",
            "489484",
            imported("experiments/logs/R-iter3-paired-gsd.log"),
            paired_outputs,
        ),
        run_from_terminal(
            "R-paired-gsd-v2",
            "n005",
            "evaluation",
            "python experiments/code/run_leakage_free_paired_gsd.py",
            "489486",
            imported("experiments/logs/R-iter3-paired-gsd-v2.log"),
            paired_v2_outputs,
        ),
        run_from_terminal(
            "R-mobilenet-replication-v1-failed",
            "n007",
            "training",
            "python experiments/code/run_mobilenet_operator_replication.py",
            "489492",
            "experiments/logs/R-mobilenet-operator-replication.log",
            [],
            status="failed",
            cwd=".",
        ),
        run_from_terminal(
            "R-mobilenet-replication-v2-failed",
            "n007",
            "training",
            "python experiments/code/run_mobilenet_operator_replication.py",
            "489493",
            "experiments/logs/R-mobilenet-operator-replication-v2.log",
            [],
            status="failed",
            cwd=".",
        ),
        run_from_terminal(
            "R-mobilenet-replication",
            "n007",
            "training-and-evaluation",
            "python experiments/code/run_mobilenet_operator_replication.py",
            "489494",
            "experiments/logs/R-mobilenet-operator-replication-v3.log",
            mobile_outputs,
            cwd=".",
        ),
        run_from_terminal(
            "R-resnet-operator-sensitivity",
            "n008",
            "evaluation",
            "python experiments/code/run_resnet_operator_sensitivity.py",
            "489495",
            "experiments/logs/R-resnet-operator-sensitivity.log",
            resnet_operator_outputs,
            cwd=".",
        ),
        run_from_terminal(
            "R-reference-reveal-mask",
            "n010",
            "evaluation",
            "python experiments/code/run_reference_reveal_mask.py",
            "489496",
            "experiments/logs/R-reference-reveal-mask.log",
            reveal_mask_outputs,
            cwd=".",
        ),
        run_from_terminal(
            "R-idalia-paired-sensitivity",
            "n011",
            "evaluation",
            "python experiments/code/run_idalia_paired_sensitivity.py",
            "489498",
            "experiments/logs/R-idalia-paired-sensitivity.log",
            idalia_outputs,
            cwd=".",
        ),
        local_run(
            "R-import-evidence",
            "packaging",
            "python experiments/code/import_parent_evidence.py",
            "experiments/logs/R-import-parent-evidence.log",
            ["experiments/imported/import_manifest.json"],
            1.0,
        ),
        local_run(
            "R-literature-audit",
            "analysis",
            "python experiments/code/build_information_set_literature_audit.py",
            "experiments/logs/R-literature-audit.log",
            ["experiments/derived/information_set_literature_audit.json"],
            1.0,
        ),
        local_run(
            "R-architecture-verification",
            "analysis",
            "python experiments/code/verify_architecture_replication.py",
            "experiments/logs/R-architecture-verification.log",
            [
                "experiments/derived/architecture_replication/verification.json"
            ],
            2.0,
        ),
        local_run(
            "R-reference-reveal-mask-verification",
            "analysis",
            "python experiments/code/verify_reference_reveal_mask.py",
            "experiments/logs/R-reference-reveal-mask-verification.log",
            [
                "experiments/derived/reference_reveal_mask/verification.json"
            ],
            2.0,
        ),
        local_run(
            "R-idalia-verification",
            "analysis",
            "python experiments/code/verify_idalia_sensitivity.py",
            "experiments/logs/R-idalia-verification.log",
            ["experiments/derived/idalia_paired/verification.json"],
            1.0,
        ),
        local_run(
            "R-paired-adjudication",
            "analysis",
            "python experiments/code/adjudicate_paired_protocol.py",
            "experiments/logs/R-paired-adjudication.log",
            ["experiments/derived/paired_protocol_adjudication.json"],
            1.0,
        ),
        local_run(
            "R-per-example-verify",
            "analysis",
            "python experiments/code/verify_per_example_outputs.py",
            "experiments/logs/R-per-example-verify.log",
            ["experiments/derived/per_example_verification.json"],
            6.0,
        ),
        local_run(
            "R-gsd-metadata",
            "data-preparation",
            "python experiments/code/capture_gsd_metadata.py",
            "experiments/logs/R-gsd-metadata.log",
            [
                "experiments/derived/metadata/CRASAR_statistics.csv",
                "experiments/derived/metadata/CRASAR_gsd_records.json",
            ],
            1.0,
        ),
        local_run(
            "R-audit-artifacts",
            "analysis",
            "python experiments/code/generate_audit_artifacts.py",
            "experiments/logs/R-audit-artifacts-final.log",
            [
                "paper/tables/hash_audit.tex",
                "paper/tables/privileged_inflation.tex",
                "paper/tables/anchor_matched_sensitivity.tex",
                "paper/tables/matched_auroc.tex",
                "paper/tables/crasar_summary.tex",
                "paper/tables/measured_gsd.tex",
                "paper/tables/architecture_operator_replication.tex",
                "paper/tables/reference_reveal_mask.tex",
                "paper/tables/idalia_sensitivity.tex",
                "paper/figures/information_audit.pdf",
                "paper/figures/measured_gsd_audit.pdf",
                "paper/figures/architecture_operator_replication.pdf",
            ],
            1.0,
        ),
        local_run(
            "R-claim-verification",
            "analysis",
            "python experiments/code/verify_audit_claims.py",
            "experiments/logs/R-claim-verification.log",
            ["experiments/derived/claim_verification.json"],
            1.0,
        ),
        local_run(
            "R-environment-capture",
            "environment",
            "python experiments/code/capture_environment.py",
            "experiments/logs/R-environment-capture-v2.log",
            ["experiments/derived/environment.json"],
            4.0,
        ),
        local_run(
            "R-paper-build",
            "build",
            "tectonic main.tex --keep-logs --keep-intermediates",
            "experiments/logs/R-paper-build-final.log",
            ["paper/main.pdf"],
            4.0,
        ),
        local_run(
            "R-submission-package",
            "packaging",
            "python experiments/code/build_submission_package.py",
            "experiments/logs/R-submission-package.log",
            [
                "submission/resolution-audit-source.zip",
                "submission/resolution-audit-reproducibility.zip",
                "submission/checksums.json",
                "submission/source_manifest.json",
                "submission/reproducibility_manifest.json",
            ],
            10.0,
        ),
        local_run(
            "R-submission-verify",
            "verification",
            "python experiments/code/verify_submission_package.py",
            "experiments/logs/R-submission-verify.log",
            ["submission/verification.json"],
            5.0,
        ),
    ]
    (ROOT / "experiments/run-ledger.jsonl").write_text(
        "".join(json.dumps(run, allow_nan=False) + "\n" for run in runs)
    )

    information_path = imported(
        "experiments/derived/greatness_iteration3/information_matched_audit.json"
    )
    crasar_path = imported(
        "experiments/derived/greatness_iteration3/"
        "crasar/leakage_free_crasar.json"
    )
    paired_path = imported(
        "experiments/derived/greatness_iteration3/"
        "crasar/leakage_free_paired_gsd.json"
    )
    paired_v2_path = imported(
        "experiments/derived/greatness_iteration3/"
        "crasar/leakage_free_paired_gsd_v2.json"
    )
    adjudication_path = "experiments/derived/paired_protocol_adjudication.json"
    per_example_path = "experiments/derived/per_example_verification.json"
    gsd_metadata_path = "experiments/derived/metadata/CRASAR_gsd_records.json"
    mobile_path = (
        "experiments/derived/architecture_replication/"
        "mobilenet_aider_operator_audit.json"
    )
    resnet_operator_path = (
        "experiments/derived/architecture_replication/"
        "resnet_operator_sensitivity.json"
    )
    architecture_verification_path = (
        "experiments/derived/architecture_replication/verification.json"
    )
    literature_audit_path = (
        "experiments/derived/information_set_literature_audit.json"
    )
    reveal_mask_path = (
        "experiments/derived/reference_reveal_mask/reveal_mask_summary.json"
    )
    reveal_mask_verification_path = (
        "experiments/derived/reference_reveal_mask/verification.json"
    )
    idalia_path = (
        "experiments/derived/idalia_paired/idalia_paired_sensitivity.json"
    )
    idalia_verification_path = (
        "experiments/derived/idalia_paired/verification.json"
    )
    aider_audit = imported(
        "experiments/derived/greatness_iteration3/aider_dedup_audit.json"
    )
    hurricane_audit = imported(
        "experiments/derived/greatness_iteration3/hurricane_dedup_audit.json"
    )
    claims = [
        {
            "claim_id": "C001",
            "claim": "Encoded-byte audits found two conflicting-label cross-split AIDER groups and three cross-split Hurricane groups; all members were excluded and no cross-split hash remained.",
            "type": "quantitative",
            "scope": "Corrected frozen AIDER and Hurricane splits",
            "status": "verified",
            "verified_at": NOW,
            "source_artifacts": [source(aider_audit), source(hurricane_audit)],
            "analysis_command": "python experiments/code/prepare_aider_dedup.py && python experiments/code/prepare_hurricane_dedup.py",
            "paper_locations": ["results"],
            "run_ids": ["R-aider-dedup-prep", "R-hurricane-dedup-prep"],
        },
        {
            "claim_id": "C002",
            "claim": "The descriptive reference-dependent minus received-image error-AUROC gaps are 0.341 on AIDER and 0.399 on Hurricane; all six paired 95% bootstrap intervals exclude zero.",
            "type": "quantitative",
            "scope": "Deduplicated s=8 tests, ResNet-18 seeds 101/202/303",
            "status": "verified",
            "verified_at": NOW,
            "source_artifacts": [source(information_path)],
            "analysis_command": "python experiments/code/run_information_matched_audit.py",
            "paper_locations": ["abstract", "results", "discussion"],
            "run_ids": ["R-information-audit-v3"],
        },
        {
            "claim_id": "C003",
            "claim": "Under matched received-image information, ViM has mean AUROC 0.758 on AIDER and 0.643 on Hurricane, while received consistency reaches 0.623 and 0.590; no deployable score dominates every corpus and seed.",
            "type": "quantitative",
            "scope": "Same models, received images, source training and validation information",
            "status": "verified",
            "verified_at": NOW,
            "source_artifacts": [source(information_path)],
            "analysis_command": "python experiments/code/run_information_matched_audit.py",
            "paper_locations": ["results"],
            "run_ids": ["R-information-audit-v3"],
        },
        {
            "claim_id": "C004",
            "claim": "The guarded CRASAR split contains 1,628 crops with zero cross-split rectangle intersections; mean internal accuracy is 0.794 versus majority accuracy 0.635.",
            "type": "quantitative",
            "scope": "Lancaster and Summerlin guarded 2048-pixel blocks, three seeds",
            "status": "verified",
            "verified_at": NOW,
            "source_artifacts": [source(crasar_path)],
            "analysis_command": "python experiments/code/run_leakage_free_crasar.py --epochs 12",
            "paper_locations": ["results"],
            "run_ids": ["R-crasar-train"],
        },
        {
            "claim_id": "C005",
            "claim": "The held-out measured-GSD evaluation contains 1,441 same-building pairs from four sites and two events; pooled UAS accuracy exceeds the 0.533 majority baseline for every seed.",
            "type": "quantitative",
            "scope": "CRASAR Hurricane Ian and Hurricane Michael sites",
            "status": "verified",
            "verified_at": NOW,
            "source_artifacts": [source(paired_path)],
            "analysis_command": "python experiments/code/run_leakage_free_paired_gsd.py",
            "paper_locations": ["abstract", "results"],
            "run_ids": ["R-paired-gsd"],
        },
        {
            "claim_id": "C006",
            "claim": "Received consistency has pooled satellite AUROC lift 0.149 over confidence, but both Hurricane Michael site lifts are negative and all seed-level spatial-cluster intervals include zero; W12 fails.",
            "type": "quantitative",
            "scope": "Four held-out paired sites, three model seeds",
            "status": "verified",
            "verified_at": NOW,
            "source_artifacts": [source(paired_path)],
            "analysis_command": "python experiments/code/run_leakage_free_paired_gsd.py",
            "paper_locations": ["abstract", "results", "discussion"],
            "run_ids": ["R-paired-gsd"],
        },
        {
            "claim_id": "C007",
            "claim": "At target coverage 0.70, preregistered mean-based W13 passes because received consistency has lower mean absolute coverage error (0.106 vs 0.201) and mean selective risk (0.456 vs 0.548); the post-hoc all-seed robustness rule fails at seed 202.",
            "type": "quantitative",
            "scope": "Source-validation thresholds transferred to held-out satellite pairs",
            "status": "verified",
            "verified_at": NOW,
            "source_artifacts": [
                source(paired_path),
                source(adjudication_path),
            ],
            "analysis_command": "python experiments/code/adjudicate_paired_protocol.py",
            "paper_locations": ["results", "discussion"],
            "run_ids": ["R-paired-gsd", "R-paired-adjudication"],
        },
        {
            "claim_id": "C008",
            "claim": "A post-hoc anchor-matched sensitivity retains fine-reference minus received-consistency AUROC gaps of 0.336 on AIDER and 0.398 on Hurricane, with all six paired intervals positive.",
            "type": "quantitative",
            "scope": "Same s=8 anchor and three comparisons per score; degradation direction remains different",
            "status": "verified",
            "verified_at": NOW,
            "source_artifacts": [source(information_path)],
            "analysis_command": "python experiments/code/run_information_matched_audit.py",
            "paper_locations": ["method", "results", "limitations"],
            "run_ids": ["R-information-audit-v3"],
        },
        {
            "claim_id": "C009",
            "claim": "Per-example arrays reproduce every reported AUROC and supply the preregistered ViM-minus-EO-kNN intervals and paired prediction transitions.",
            "type": "quantitative",
            "scope": "AIDER, Hurricane and paired CRASAR per-example outputs",
            "status": "verified",
            "verified_at": NOW,
            "source_artifacts": [source(per_example_path)],
            "analysis_command": "python experiments/code/verify_per_example_outputs.py",
            "paper_locations": ["results", "appendix"],
            "run_ids": [
                "R-information-audit-v3",
                "R-paired-gsd-v2",
                "R-per-example-verify",
            ],
        },
        {
            "claim_id": "C010",
            "claim": "Post-hoc joint UAS/satellite overlap clustering leaves two effective clusters at Harlem and one at each other acquisition; pooled AUROC-lift intervals still include zero.",
            "type": "quantitative",
            "scope": "Four measured-GSD acquisitions; connected overlap components",
            "status": "verified",
            "verified_at": NOW,
            "source_artifacts": [source(paired_v2_path)],
            "analysis_command": "python experiments/code/run_leakage_free_paired_gsd.py",
            "paper_locations": ["method", "results", "appendix"],
            "run_ids": ["R-paired-gsd-v2"],
        },
        {
            "claim_id": "C011",
            "claim": "The exact CRASAR statistics.csv used for all GSD values has SHA-256 b2a2fee6b1a631e2f2dc5f24a98988f3152728fd5dd793fc62b7707271a38936.",
            "type": "provenance",
            "scope": "CRASAR revision 47cf4ab3a94d42978975f7d23338a996125ac0e9",
            "status": "verified",
            "verified_at": NOW,
            "source_artifacts": [source(gsd_metadata_path)],
            "analysis_command": "python experiments/code/capture_gsd_metadata.py",
            "paper_locations": ["method"],
            "run_ids": ["R-gsd-metadata"],
        },
        {
            "claim_id": "C012",
            "claim": "The anchor-matched fine-reference minus received-image AUROC gap is positive across ResNet-18 and MobileNetV3-Small, four degradation operators, two corpora where available, and all 36 seed-level intervals; M1-M3 pass.",
            "type": "quantitative",
            "scope": "AIDER MobileNet and ResNet; Hurricane ResNet; seeds 101/202/303; bicubic/bilinear/nearest/box",
            "status": "verified",
            "verified_at": NOW,
            "source_artifacts": [
                source(mobile_path),
                source(resnet_operator_path),
                source(architecture_verification_path),
            ],
            "analysis_command": "python experiments/code/verify_architecture_replication.py",
            "paper_locations": ["abstract", "experimental-setup", "results", "discussion"],
            "run_ids": [
                "R-mobilenet-replication",
                "R-resnet-operator-sensitivity",
                "R-architecture-verification",
            ],
        },
        {
            "claim_id": "C013",
            "claim": "A structured scoping audit of 12 primary works found no exact unavailable-fine-reference deployment comparison in the bounded literature set.",
            "type": "literature-audit",
            "scope": "Six documented search queries across disagreement, TTA, EO OOD, multi-resolution EO and spatial validation",
            "status": "verified",
            "verified_at": NOW,
            "source_artifacts": [source(literature_audit_path)],
            "analysis_command": "python experiments/code/build_information_set_literature_audit.py",
            "paper_locations": ["related-work", "appendix"],
            "run_ids": ["R-literature-audit"],
        },
        {
            "claim_id": "C014",
            "claim": "Fine-reference reveal/mask identification passes all 24 ResNet corpus/seed/operator combinations; the minimum aligned-minus-masked mean AUROC gap is 0.359 and the maximum empirical probability is 0.0099.",
            "type": "quantitative",
            "scope": "AIDER and Hurricane, ResNet-18 seeds 101/202/303, four operators, 100 correspondence permutations",
            "status": "verified",
            "verified_at": NOW,
            "source_artifacts": [
                source(reveal_mask_path),
                source(reveal_mask_verification_path),
            ],
            "analysis_command": "python experiments/code/verify_reference_reveal_mask.py",
            "paper_locations": ["abstract", "method", "results", "discussion", "conclusion"],
            "run_ids": [
                "R-reference-reveal-mask",
                "R-reference-reveal-mask-verification",
            ],
        },
        {
            "claim_id": "C015",
            "claim": "The third-event Hurricane Idalia sensitivity contains 458 UAS/crewed pairs in 37 joint clusters; E1/E2 pass and mean received-consistency AUROC lift is 0.325 with positive cluster intervals for all seeds.",
            "type": "quantitative",
            "scope": "Held-out Steinhatchee River, CRASAR UAS 12.70 cm/px and crewed 15.02 cm/px, three frozen ResNet seeds",
            "status": "verified",
            "verified_at": NOW,
            "source_artifacts": [
                source(idalia_path),
                source(idalia_verification_path),
            ],
            "analysis_command": "python experiments/code/verify_idalia_sensitivity.py",
            "paper_locations": ["abstract", "experimental-setup", "results", "discussion", "limitations"],
            "run_ids": [
                "R-idalia-paired-sensitivity",
                "R-idalia-verification",
            ],
        },
    ]
    (ROOT / "evidence/claim-ledger.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    **claim,
                    "run_ids": claim["run_ids"]
                    + (
                        ["R-claim-verification"]
                        if int(claim["claim_id"][1:]) <= 7
                        else []
                    ),
                },
                allow_nan=False,
            )
            + "\n"
            for claim in claims
        )
    )

    publication = [
        (
            "A-table-hash",
            "table",
            "paper/tables/hash_audit.tex",
            [aider_audit, hurricane_audit],
            ["C001"],
        ),
        (
            "A-table-inflation",
            "table",
            "paper/tables/privileged_inflation.tex",
            [information_path],
            ["C002"],
        ),
        (
            "A-table-anchor-matched",
            "table",
            "paper/tables/anchor_matched_sensitivity.tex",
            [information_path],
            ["C008"],
        ),
        (
            "A-table-architecture-operator",
            "table",
            "paper/tables/architecture_operator_replication.tex",
            [mobile_path, resnet_operator_path],
            ["C012"],
        ),
        (
            "A-table-reveal-mask",
            "table",
            "paper/tables/reference_reveal_mask.tex",
            [reveal_mask_path],
            ["C014"],
        ),
        (
            "A-table-idalia",
            "table",
            "paper/tables/idalia_sensitivity.tex",
            [idalia_path],
            ["C015"],
        ),
        (
            "A-table-matched",
            "table",
            "paper/tables/matched_auroc.tex",
            [information_path],
            ["C002", "C003"],
        ),
        (
            "A-figure-information",
            "figure",
            "paper/figures/information_audit.pdf",
            [information_path],
            ["C002", "C003"],
        ),
        (
            "A-figure-architecture-operator",
            "figure",
            "paper/figures/architecture_operator_replication.pdf",
            [mobile_path, resnet_operator_path],
            ["C012"],
        ),
        (
            "A-table-crasar",
            "table",
            "paper/tables/crasar_summary.tex",
            [crasar_path, paired_path],
            ["C004", "C005", "C006"],
        ),
        (
            "A-table-measured",
            "table",
            "paper/tables/measured_gsd.tex",
            [paired_path],
            ["C005", "C006"],
        ),
        (
            "A-figure-measured",
            "figure",
            "paper/figures/measured_gsd_audit.pdf",
            [paired_path],
            ["C005", "C006", "C007"],
        ),
    ]
    artifacts = []
    for artifact_id, artifact_type, path, sources, claim_ids in publication:
        artifacts.append(
            {
                "artifact_id": artifact_id,
                "type": artifact_type,
                **record(path),
                "generator_code": "experiments/code/generate_audit_artifacts.py",
                "generator_command": "python experiments/code/generate_audit_artifacts.py",
                "justification": "Programmatically generated from registered corrective result JSON.",
                "latex_reference": path.removeprefix("paper/"),
                "verified_at": NOW,
                "run_ids": ["R-audit-artifacts"],
                "source_artifacts": [source(item) for item in sources],
                "claim_ids": claim_ids,
            }
        )
    for artifact in [
        {
            "artifact_id": "A-paired-adjudication",
            "type": "algorithm-output",
            **record(adjudication_path),
            "generator_code": "experiments/code/adjudicate_paired_protocol.py",
            "generator_command": "python experiments/code/adjudicate_paired_protocol.py",
            "justification": "Corrects W13 against the frozen mean-based rule and preserves the post-hoc all-seed sensitivity.",
            "latex_reference": "paired_protocol_adjudication.json",
            "verified_at": NOW,
            "run_ids": ["R-paired-adjudication"],
            "source_artifacts": [source(paired_path)],
            "claim_ids": ["C007"],
        },
        {
            "artifact_id": "A-per-example-verification",
            "type": "algorithm-output",
            **record(per_example_path),
            "generator_code": "experiments/code/verify_per_example_outputs.py",
            "generator_command": "python experiments/code/verify_per_example_outputs.py",
            "justification": "Recomputes AUROCs, omitted paired baseline intervals, transitions, and site sensitivities from portable arrays.",
            "latex_reference": "per_example_verification.json",
            "verified_at": NOW,
            "run_ids": ["R-per-example-verify"],
            "source_artifacts": [source(information_path), source(paired_v2_path)],
            "claim_ids": ["C009", "C010"],
        },
        {
            "artifact_id": "A-gsd-metadata",
            "type": "other",
            **record(gsd_metadata_path),
            "generator_code": "experiments/code/capture_gsd_metadata.py",
            "generator_command": "python experiments/code/capture_gsd_metadata.py",
            "justification": "Hashes the exact fixed-revision metadata rows used for all measured GSD values.",
            "latex_reference": "CRASAR_gsd_records.json",
            "verified_at": NOW,
            "run_ids": ["R-gsd-metadata"],
            "source_artifacts": [
                source("experiments/derived/metadata/CRASAR_statistics.csv")
            ],
            "claim_ids": ["C011"],
        },
        {
            "artifact_id": "A-architecture-verification",
            "type": "algorithm-output",
            **record(architecture_verification_path),
            "generator_code": "experiments/code/verify_architecture_replication.py",
            "generator_command": "python experiments/code/verify_architecture_replication.py",
            "justification": "Recomputes 72 raw AUROCs and verifies M1-M3 across architectures and operators.",
            "latex_reference": "architecture_replication_verification.json",
            "verified_at": NOW,
            "run_ids": ["R-architecture-verification"],
            "source_artifacts": [source(mobile_path), source(resnet_operator_path)],
            "claim_ids": ["C012"],
        },
        {
            "artifact_id": "A-information-set-literature-audit",
            "type": "appendix",
            **record(literature_audit_path),
            "generator_code": "experiments/code/build_information_set_literature_audit.py",
            "generator_command": "python experiments/code/build_information_set_literature_audit.py",
            "justification": "Records search queries, included works, information sets, and bounded prevalence result.",
            "latex_reference": "information_set_literature_audit.json",
            "verified_at": NOW,
            "run_ids": ["R-literature-audit"],
            "source_artifacts": [source("research/literature.jsonl")],
            "claim_ids": ["C013"],
        },
        {
            "artifact_id": "A-reference-reveal-mask-verification",
            "type": "algorithm-output",
            **record(reveal_mask_verification_path),
            "generator_code": "experiments/code/verify_reference_reveal_mask.py",
            "generator_command": "python experiments/code/verify_reference_reveal_mask.py",
            "justification": "Recomputes 2,448 aligned and masked permutation metrics and verifies M4.",
            "latex_reference": "reference_reveal_mask_verification.json",
            "verified_at": NOW,
            "run_ids": ["R-reference-reveal-mask-verification"],
            "source_artifacts": [source(reveal_mask_path)],
            "claim_ids": ["C014"],
        },
        {
            "artifact_id": "A-idalia-verification",
            "type": "algorithm-output",
            **record(idalia_verification_path),
            "generator_code": "experiments/code/verify_idalia_sensitivity.py",
            "generator_command": "python experiments/code/verify_idalia_sensitivity.py",
            "justification": "Recomputes third-event balanced accuracy and error-ranking metrics from raw arrays.",
            "latex_reference": "idalia_verification.json",
            "verified_at": NOW,
            "run_ids": ["R-idalia-verification"],
            "source_artifacts": [source(idalia_path)],
            "claim_ids": ["C015"],
        },
    ]:
        artifacts.append(artifact)
    for artifact_id, path, justification, sources in [
        (
            "A-source-archive",
            "submission/resolution-audit-source.zip",
            "Self-contained IJRS LaTeX source, local class/style, tables and figures.",
            ["paper/main.tex", "paper/figures/information_audit.pdf"],
        ),
        (
            "A-reproducibility-archive",
            "submission/resolution-audit-reproducibility.zip",
            "Submission supplement containing code, checkpoints, results, logs and ledgers.",
            ["experiments/imported/import_manifest.json", "paper/main.pdf"],
        ),
        (
            "A-package-verification",
            "submission/verification.json",
            "Manifest verification and clean-build result for both submission archives.",
            [
                "submission/resolution-audit-source.zip",
                "submission/resolution-audit-reproducibility.zip",
            ],
        ),
    ]:
        artifacts.append(
            {
                "artifact_id": artifact_id,
                "type": "other",
                **record(path),
                "generator_code": (
                    "experiments/code/verify_submission_package.py"
                    if artifact_id == "A-package-verification"
                    else "experiments/code/build_submission_package.py"
                ),
                "generator_command": (
                    "python experiments/code/verify_submission_package.py"
                    if artifact_id == "A-package-verification"
                    else "python experiments/code/build_submission_package.py"
                ),
                "justification": justification,
                "latex_reference": Path(path).name,
                "verified_at": NOW,
                "run_ids": [
                    "R-submission-verify"
                    if artifact_id == "A-package-verification"
                    else "R-submission-package"
                ],
                "source_artifacts": [source(item) for item in sources],
                "claim_ids": [],
            }
        )
    (ROOT / "evidence/artifact-ledger.jsonl").write_text(
        "".join(
            json.dumps(artifact, allow_nan=False) + "\n"
            for artifact in artifacts
        )
    )
    print("AUDIT_EVIDENCE_REGISTRATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
