#!/usr/bin/env python3
"""Import corrective experiment evidence from the pivoted parent study."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT.parent / "disaster-rcg-gsd"
DESTINATION = ROOT / "experiments/imported"
TEMPLATE_FILES = {
    "paper-ijrs/interact.cls": ROOT / "paper/interact.cls",
    "paper-ijrs/tfcad.bst": ROOT / "paper/tfcad.bst",
}
FILES = [
    "research/preregistration-deployable-rcg.md",
    "research/preregistration-leakage-free-replication.md",
    "research/preregistration-leakage-free-replication-v2.md",
    "experiments/code/prepare_aider_dedup.py",
    "experiments/code/prepare_hurricane_dedup.py",
    "experiments/code/run_primary_multiseed.py",
    "experiments/code/run_information_matched_audit.py",
    "experiments/code/run_leakage_free_crasar.py",
    "experiments/code/run_leakage_free_paired_gsd.py",
    "experiments/derived/greatness_strengthening/deployable_rcg.json",
    "experiments/derived/greatness_iteration3/aider_splits_dedup.json",
    "experiments/derived/greatness_iteration3/aider_dedup_audit.json",
    "experiments/derived/greatness_iteration3/hurricane_splits_dedup.json",
    "experiments/derived/greatness_iteration3/hurricane_dedup_audit.json",
    "experiments/derived/greatness_iteration3/primary_multiseed.json",
    "experiments/derived/greatness_iteration3/information_matched_audit.json",
    "experiments/derived/greatness_iteration3/crasar/patch_manifest.json",
    "experiments/derived/greatness_iteration3/crasar/split_audit.json",
    "experiments/derived/greatness_iteration3/crasar/leakage_free_crasar.json",
    "experiments/derived/greatness_iteration3/crasar/leakage_free_paired_gsd.json",
    "experiments/derived/greatness_iteration3/crasar/leakage_free_paired_gsd_v2.json",
    "experiments/derived/greatness_iteration3/crasar/paired_evaluation/"
    "harlem-heights/pair_manifest.json",
    "experiments/derived/greatness_iteration3/crasar/paired_evaluation/"
    "mcgregor-college-parkway-south-1/pair_manifest.json",
    "experiments/derived/greatness_iteration3/crasar/paired_evaluation/"
    "mexico-beach-2018-10-13/pair_manifest.json",
    "experiments/derived/greatness_iteration3/crasar/paired_evaluation/"
    "mexico-beach-2018-10-14/pair_manifest.json",
    "experiments/logs/R-iter3-train-aider.log",
    "experiments/logs/R-iter3-train-hurricane.log",
    "experiments/logs/R-iter3-aider-dedup-prep.log",
    "experiments/logs/R-iter3-hurricane-dedup-prep.log",
    "experiments/logs/R-iter3-information-audit.log",
    "experiments/logs/R-iter3-information-audit-v2.log",
    "experiments/logs/R-iter3-information-audit-v3.log",
    "experiments/logs/R-iter3-crasar-prepare.log",
    "experiments/logs/R-iter3-crasar-prepare-v2.log",
    "experiments/logs/R-iter3-train-crasar.log",
    "experiments/logs/R-iter3-mexico-download.log",
    "experiments/logs/R-iter3-paired-gsd.log",
    "experiments/logs/R-iter3-paired-gsd-v2.log",
]
for corpus in ("aider", "hurricane"):
    for seed in (101, 202, 303):
        FILES.extend(
            [
                f"experiments/derived/greatness_iteration3/{corpus}/"
                f"seed_{seed}/best.pt",
                f"experiments/derived/greatness_iteration3/{corpus}/"
                f"seed_{seed}/history.json",
                f"experiments/derived/greatness_iteration3/{corpus}/"
                f"seed_{seed}/metrics.json",
            ]
        )
for seed in (101, 202, 303):
    FILES.extend(
        [
            f"experiments/derived/greatness_iteration3/crasar/"
            f"seed_{seed}/best.pt",
            f"experiments/derived/greatness_iteration3/crasar/"
            f"seed_{seed}/history.json",
            f"experiments/derived/greatness_iteration3/crasar/"
            f"seed_{seed}/metrics.json",
        ]
    )
    FILES.append(
        f"experiments/derived/greatness_iteration3/information_scores/"
        f"aider_seed_{seed}.npz"
    )
    FILES.append(
        f"experiments/derived/greatness_iteration3/information_scores/"
        f"hurricane_seed_{seed}.npz"
    )
    FILES.append(
        f"experiments/derived/greatness_iteration3/crasar/paired_scores/"
        f"seed_{seed}.npz"
    )
FILES.append(
    "experiments/derived/greatness_iteration3/crasar/"
    "paired_scores/paired_score_index.json"
)
for site in (
    "harlem-heights",
    "mcgregor-college-parkway-south-1",
    "mexico-beach-2018-10-13",
    "mexico-beach-2018-10-14",
):
    FILES.append(
        "experiments/derived/greatness_iteration3/crasar/"
        f"paired_evaluation/{site}/selection_audit.json"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_string(value: str) -> str:
    if not value.startswith("/"):
        return value
    if "/AIDER/" in value:
        return value.split("/AIDER/", 1)[1]
    marker = "/experiments/derived/greatness_iteration3/crasar/"
    if marker in value:
        return value.split(marker, 1)[1]
    if "/datasets--CRASAR--CRASAR-U-DROIDs/" in value:
        tail = value.split("/snapshots/", 1)[-1]
        revision, relative = tail.split("/", 1)
        return (
            "hf://datasets/CRASAR/CRASAR-U-DROIDs@"
            f"{revision}/{relative}"
        )
    return Path(value).name


def sanitize_json(value):
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, str):
        return portable_string(value)
    return value


def make_json_portable(path: Path) -> None:
    if path.suffix != ".json":
        return
    try:
        payload = json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return
    path.write_text(json.dumps(sanitize_json(payload), indent=2) + "\n")


def main() -> None:
    manifest = []
    for relative, destination in TEMPLATE_FILES.items():
        source = PARENT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest.append(
            {
                "source": f"parent-study/{relative}",
                "path": str(destination.relative_to(ROOT)),
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
            }
        )
    for relative in FILES:
        source = PARENT / relative
        if not source.exists():
            raise FileNotFoundError(source)
        destination = DESTINATION / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        make_json_portable(destination)
        manifest.append(
            {
                "source": f"parent-study/{relative}",
                "path": str(destination.relative_to(ROOT)),
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
            }
        )
    output = DESTINATION / "import_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print("PARENT_EVIDENCE_IMPORT_COMPLETE", len(manifest), output, flush=True)


if __name__ == "__main__":
    main()
