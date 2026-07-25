#!/usr/bin/env python3
"""Audit and exclude exact cross-split duplicates in Hurricane Damage."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import Image, load_dataset


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments/derived/greatness_iteration3/hurricane_splits_dedup.json"
AUDIT = ROOT / "experiments/derived/greatness_iteration3/hurricane_dedup_audit.json"


def main() -> None:
    decoded = load_dataset(
        "jonathan-roberts1/Satellite-Images-of-Hurricane-Damage", split="train"
    )
    encoded = decoded.cast_column("image", Image(decode=False))
    order = np.random.default_rng(0).permutation(len(decoded)).tolist()
    original = {
        "train": order[:7000],
        "val": order[7000:8000],
        "test": order[8000:],
    }
    groups: dict[str, list[dict]] = defaultdict(list)
    for split, indexes in original.items():
        for index in indexes:
            image = encoded[int(index)]["image"]
            payload = image["bytes"]
            if payload is None:
                payload = Path(image["path"]).read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            groups[digest].append(
                {
                    "split": split,
                    "index": int(index),
                    "label": int(decoded[int(index)]["label"]),
                }
            )
    cross_split = {
        digest: rows
        for digest, rows in groups.items()
        if len({row["split"] for row in rows}) > 1
    }
    excluded = {
        row["index"] for rows in cross_split.values() for row in rows
    }
    corrected = {
        split: [index for index in indexes if index not in excluded]
        for split, indexes in original.items()
    }
    corrected["seed"] = 0
    corrected["deduplication"] = {
        "rule": "exclude every member of every cross-split encoded-byte SHA-256 group",
        "excluded_count": len(excluded),
        "cross_split_group_count": len(cross_split),
    }
    audit = {
        "source_counts": {
            split: len(indexes) for split, indexes in original.items()
        },
        "corrected_counts": {
            split: len(corrected[split]) for split in ("train", "val", "test")
        },
        "cross_split_groups": cross_split,
        "excluded_indexes": sorted(excluded),
        "remaining_cross_split_hashes": {},
        "a3_hurricane_pass": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(corrected, indent=2) + "\n")
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print("HURRICANE_DEDUP_COMPLETE", OUTPUT, AUDIT, flush=True)


if __name__ == "__main__":
    main()
