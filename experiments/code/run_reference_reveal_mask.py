#!/usr/bin/env python3
"""Reveal/mask experiment isolating corresponding finer-view information."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "imported/experiments/code")
)
import run_primary_multiseed as base

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_mobilenet_operator_replication as helper
from run_resnet_operator_sensitivity import OperatorItems, corpus_data


ROOT = Path(__file__).resolve().parents[2]
IMPORTED = ROOT / "experiments/imported/experiments/derived/greatness_iteration3"
OUTPUT = ROOT / "experiments/derived/reference_reveal_mask"
SEEDS = (101, 202, 303)
SCALES = (1, 2, 4, 8)
PERMUTATIONS = 100
PERMUTATION_SEED = 260725


def evaluate(corpus, seed, operator, device):
    (
        _,
        _,
        test_rows,
        image_getter,
        label_getter,
        _,
        _,
        num_labels,
    ) = corpus_data(corpus)
    checkpoint = IMPORTED / f"{corpus}/seed_{seed}/best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = base.build_model(num_labels)
    model.load_state_dict(state["state_dict"])
    model.to(device)
    ladder = {}
    labels = None
    for scale in SCALES:
        ladder[scale], labels, _ = base.predict(
            model,
            OperatorItems(
                test_rows, image_getter, label_getter, scale, operator
            ),
            device,
        )
    prediction = ladder[8].argmax(1)
    errors = (prediction != labels).astype(int)
    aligned = np.mean(
        [helper.js(ladder[8], ladder[scale]) for scale in (1, 2, 4)],
        axis=0,
    )
    aligned_auc = float(roc_auc_score(errors, aligned))
    rng = np.random.default_rng(
        PERMUTATION_SEED
        + seed
        + list(helper.OPERATORS).index(operator) * 10_000
        + (0 if corpus == "aider" else 100_000)
    )
    masked_scores = []
    masked_aurocs = []
    for _ in range(PERMUTATIONS):
        permutation = rng.permutation(len(errors))
        masked = np.mean(
            [
                helper.js(ladder[8], ladder[scale][permutation])
                for scale in (1, 2, 4)
            ],
            axis=0,
        )
        masked_scores.append(masked)
        masked_aurocs.append(float(roc_auc_score(errors, masked)))
    masked_scores = np.asarray(masked_scores, dtype=np.float32)
    masked_aurocs = np.asarray(masked_aurocs)
    probability = float(
        (1 + np.sum(masked_aurocs >= aligned_auc)) / (PERMUTATIONS + 1)
    )
    score_root = OUTPUT / corpus / f"seed_{seed}"
    score_root.mkdir(parents=True, exist_ok=True)
    score_path = score_root / f"{operator}.npz"
    np.savez_compressed(
        score_path,
        labels=labels.astype(np.int64),
        predictions=prediction.astype(np.int64),
        errors=errors.astype(np.int8),
        aligned_score=aligned,
        masked_scores=masked_scores,
        masked_aurocs=masked_aurocs,
    )
    return {
        "n": len(errors),
        "n_errors": int(errors.sum()),
        "aligned_auroc": aligned_auc,
        "masked_auroc": {
            "mean": float(masked_aurocs.mean()),
            "std": float(masked_aurocs.std(ddof=1)),
            "min": float(masked_aurocs.min()),
            "max": float(masked_aurocs.max()),
        },
        "aligned_minus_masked_mean": float(
            aligned_auc - masked_aurocs.mean()
        ),
        "empirical_probability": probability,
        "pass": (
            aligned_auc > masked_aurocs.mean() and probability <= 0.05
        ),
        "score_path": str(score_path.relative_to(ROOT)),
    }


def main():
    started = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    results = {}
    for corpus in ("aider", "hurricane"):
        results[corpus] = {}
        for seed in SEEDS:
            results[corpus][str(seed)] = {}
            for operator in helper.OPERATORS:
                result = evaluate(corpus, seed, operator, device)
                results[corpus][str(seed)][operator] = result
                print(
                    "REVEAL_MASK_COMPLETE",
                    corpus,
                    seed,
                    operator,
                    json.dumps(result),
                    flush=True,
                )
    rows = [
        result
        for corpus in results.values()
        for seed in corpus.values()
        for result in seed.values()
    ]
    summary = {
        "protocol": {
            "corpora": ["aider", "hurricane"],
            "architecture": "microsoft/resnet-18",
            "seeds": list(SEEDS),
            "operators": list(helper.OPERATORS),
            "scales": list(SCALES),
            "permutations": PERMUTATIONS,
            "permutation_seed": PERMUTATION_SEED,
            "device": str(device),
        },
        "results": results,
        "aggregate": {
            "combinations": len(rows),
            "passing_combinations": sum(row["pass"] for row in rows),
            "minimum_gap": min(
                row["aligned_minus_masked_mean"] for row in rows
            ),
            "maximum_empirical_probability": max(
                row["empirical_probability"] for row in rows
            ),
            "m4_reveal_mask_pass": all(row["pass"] for row in rows),
        },
        "runtime_seconds": time.time() - started,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT / "reveal_mask_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print("REFERENCE_REVEAL_MASK_COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
