#!/usr/bin/env python3
"""Exclude conflicting-label exact duplicates from the frozen AIDER split."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/derived/aider_rcg/aider_splits.json"
OUTPUT = ROOT / "experiments/derived/greatness_iteration3/aider_splits_dedup.json"
AUDIT = ROOT / "experiments/derived/greatness_iteration3/aider_dedup_audit.json"


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    splits = json.loads(SOURCE.read_text())
    groups: dict[str, list[dict]] = defaultdict(list)
    for split in ("train", "val", "test"):
        for item in splits[split]:
            groups[sha256(item["path"])].append(
                {
                    "split": split,
                    "path": item["path"],
                    "label": int(item["label"]),
                    "class": item["class"],
                }
            )

    conflicting = {
        digest: rows
        for digest, rows in groups.items()
        if len({row["label"] for row in rows}) > 1
    }
    excluded_paths = {
        row["path"] for rows in conflicting.values() for row in rows
    }
    corrected = {
        **{key: value for key, value in splits.items() if key not in {"train", "val", "test"}},
        "train": [
            row for row in splits["train"] if row["path"] not in excluded_paths
        ],
        "val": [row for row in splits["val"] if row["path"] not in excluded_paths],
        "test": [
            row for row in splits["test"] if row["path"] not in excluded_paths
        ],
        "deduplication": {
            "rule": "exclude every member of each conflicting-label SHA-256 group",
            "source_split": str(SOURCE),
            "excluded_count": len(excluded_paths),
            "conflicting_group_count": len(conflicting),
        },
    }

    remaining: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "val", "test"):
        for item in corrected[split]:
            remaining[sha256(item["path"])].add(split)
    cross_split = {
        digest: sorted(split_names)
        for digest, split_names in remaining.items()
        if len(split_names) > 1
    }
    if cross_split:
        raise RuntimeError(f"cross-split hashes remain: {cross_split}")

    audit = {
        "source_counts": {
            split: len(splits[split]) for split in ("train", "val", "test")
        },
        "corrected_counts": {
            split: len(corrected[split]) for split in ("train", "val", "test")
        },
        "conflicting_groups": conflicting,
        "excluded_paths": sorted(excluded_paths),
        "remaining_cross_split_hashes": cross_split,
        "w10_aider_pass": not cross_split,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(corrected, indent=2) + "\n")
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print("AIDER_DEDUP_COMPLETE", OUTPUT, AUDIT, flush=True)


if __name__ == "__main__":
    main()
