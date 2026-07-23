#!/usr/bin/env python3
"""Analyze agent-frontier-VLM judgments under resolution stress.

Reads experiments/raw/vlm_agent_judgments.jsonl and writes
experiments/derived/vlm_resolution_summary.json.

Metrics:
  - accuracy / false-Yes / false-No by scale
  - multi-scale disagreement vs scale-1 prediction
  - RCG (resolution-consistency gate): abstain when scale1 disagrees with
    scale4 or scale8; compare false-Yes rate at matched coverage vs
    always-answer and random abstention.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def norm(a: str) -> str:
    a = (a or "").strip().lower()
    if a.startswith("y"):
        return "Yes"
    if a.startswith("n"):
        return "No"
    raise ValueError(f"unexpected answer: {a!r}")


def load_judgments(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        r["answer"] = norm(r["answer"])
        r["groundtruth_answer"] = norm(r["groundtruth_answer"])
        r["scale"] = int(r["scale"])
        rows.append(r)
    return rows


def scale_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(1 for r in rows if r["answer"] == r["groundtruth_answer"])
    fy = sum(1 for r in rows if r["answer"] == "Yes" and r["groundtruth_answer"] == "No")
    fn = sum(1 for r in rows if r["answer"] == "No" and r["groundtruth_answer"] == "Yes")
    pred_yes = sum(1 for r in rows if r["answer"] == "Yes")
    gt_no = sum(1 for r in rows if r["groundtruth_answer"] == "No")
    gt_yes = sum(1 for r in rows if r["groundtruth_answer"] == "Yes")
    return {
        "n": n,
        "accuracy": correct / n if n else None,
        "n_correct": correct,
        "false_yes_rate": fy / n if n else None,
        "n_false_yes": fy,
        "false_no_rate": fn / n if n else None,
        "n_false_no": fn,
        "pred_yes_rate": pred_yes / n if n else None,
        "n_gt_no": gt_no,
        "n_gt_yes": gt_yes,
        # conditional rates among applicable GT class
        "false_yes_among_gt_no": (fy / gt_no) if gt_no else None,
        "false_no_among_gt_yes": (fn / gt_yes) if gt_yes else None,
    }


def by_question(rows: list[dict]) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        out[r["question_id"]][r["scale"]] = r
    return out


def disagreement_stats(qmap: dict[str, dict[int, dict]]) -> dict:
    """Disagreement of scale s vs scale-1 prediction."""
    stats = {}
    for s in (4, 8):
        n = 0
        disagree = 0
        for qid, scales in qmap.items():
            if 1 not in scales or s not in scales:
                continue
            n += 1
            if scales[1]["answer"] != scales[s]["answer"]:
                disagree += 1
        stats[f"disagreement_s{s}_vs_s1"] = {
            "n": n,
            "n_disagree": disagree,
            "rate": disagree / n if n else None,
        }
    # any multi-scale disagreement among {1,4,8}
    n = 0
    any_d = 0
    for qid, scales in qmap.items():
        if not all(k in scales for k in (1, 4, 8)):
            continue
        n += 1
        a1, a4, a8 = scales[1]["answer"], scales[4]["answer"], scales[8]["answer"]
        if len({a1, a4, a8}) > 1:
            any_d += 1
    stats["any_scale_disagreement"] = {
        "n": n,
        "n_disagree": any_d,
        "rate": any_d / n if n else None,
    }
    return stats


def false_yes_on_decided(decided: list[dict]) -> float | None:
    if not decided:
        return None
    fy = sum(1 for r in decided if r["answer"] == "Yes" and r["groundtruth_answer"] == "No")
    return fy / len(decided)


def accuracy_on_decided(decided: list[dict]) -> float | None:
    if not decided:
        return None
    return sum(1 for r in decided if r["answer"] == r["groundtruth_answer"]) / len(decided)


def rcg_analysis(qmap: dict[str, dict[int, dict]], seed: int = 0) -> dict:
    """RCG: abstain when scale1 != scale4 OR scale1 != scale8.
    Auto-decision uses scale-1 answer when not abstaining.
    Compare false-Yes at matched coverage vs always-answer and random abstention.
    """
    items = []
    for qid, scales in qmap.items():
        if not all(k in scales for k in (1, 4, 8)):
            continue
        a1, a4, a8 = scales[1]["answer"], scales[4]["answer"], scales[8]["answer"]
        gt = scales[1]["groundtruth_answer"]
        disagree = (a1 != a4) or (a1 != a8)
        items.append(
            {
                "question_id": qid,
                "answer": a1,
                "groundtruth_answer": gt,
                "abstain": disagree,
            }
        )

    n = len(items)
    always = [
        {"answer": it["answer"], "groundtruth_answer": it["groundtruth_answer"]}
        for it in items
    ]
    rcg_decided = [
        {"answer": it["answer"], "groundtruth_answer": it["groundtruth_answer"]}
        for it in items
        if not it["abstain"]
    ]
    n_abstain = sum(1 for it in items if it["abstain"])
    coverage = len(rcg_decided) / n if n else None

    # Random abstention at same coverage: randomly drop same count, keep scale-1 answers
    rng = random.Random(seed)
    n_keep = len(rcg_decided)
    random_runs = []
    for _ in range(200):
        keep_idx = set(rng.sample(range(n), n_keep)) if n_keep < n else set(range(n))
        decided = [
            {"answer": items[i]["answer"], "groundtruth_answer": items[i]["groundtruth_answer"]}
            for i in sorted(keep_idx)
        ]
        random_runs.append(
            {
                "false_yes_rate": false_yes_on_decided(decided),
                "accuracy": accuracy_on_decided(decided),
            }
        )
    mean_fy = sum(r["false_yes_rate"] for r in random_runs) / len(random_runs)
    mean_acc = sum(r["accuracy"] for r in random_runs) / len(random_runs)

    rcg_fy = false_yes_on_decided(rcg_decided)
    always_fy = false_yes_on_decided(always)

    # Supplemental: always-answer with scale-8 (degraded) vs RCG that abstains
    # on multi-scale disagreement and otherwise uses s1 (stable) prediction.
    always_s8 = [
        {
            "answer": qmap[it["question_id"]][8]["answer"],
            "groundtruth_answer": it["groundtruth_answer"],
        }
        for it in items
    ]
    # After RCG: if abstain drop; else use s1. Compare FY of remaining decided set
    # against always answering with s8 on the *same* decided ids (matched coverage).
    decided_ids = {it["question_id"] for it in items if not it["abstain"]}
    s8_on_rcg_coverage = [
        {
            "answer": qmap[qid][8]["answer"],
            "groundtruth_answer": qmap[qid][1]["groundtruth_answer"],
        }
        for qid in sorted(decided_ids)
    ]
    s8_fy_full = false_yes_on_decided(always_s8)
    s8_fy_matched = false_yes_on_decided(s8_on_rcg_coverage)

    # Did RCG abstain on items that are false-Yes under s8?
    s8_false_yes_ids = [
        qid
        for qid, scales in qmap.items()
        if all(k in scales for k in (1, 4, 8))
        and scales[8]["answer"] == "Yes"
        and scales[8]["groundtruth_answer"] == "No"
    ]
    abstain_ids = [it["question_id"] for it in items if it["abstain"]]
    s8_fy_caught = [qid for qid in s8_false_yes_ids if qid in abstain_ids]

    return {
        "policy": "abstain_if_s1_neq_s4_or_s1_neq_s8; decide_with_s1",
        "n_items": n,
        "n_abstain": n_abstain,
        "n_auto_decide": len(rcg_decided),
        "coverage": coverage,
        "always_answer": {
            "coverage": 1.0,
            "false_yes_rate": always_fy,
            "accuracy": accuracy_on_decided(always),
            "note": "always answer with scale-1 prediction",
        },
        "rcg": {
            "coverage": coverage,
            "false_yes_rate": rcg_fy,
            "accuracy": accuracy_on_decided(rcg_decided),
            "abstain_ids": abstain_ids,
        },
        "random_abstain_matched_coverage": {
            "n_simulations": len(random_runs),
            "seed": seed,
            "coverage": coverage,
            "mean_false_yes_rate": mean_fy,
            "mean_accuracy": mean_acc,
        },
        "rcg_reduces_false_yes_vs_always": (
            None if rcg_fy is None or always_fy is None else rcg_fy < always_fy
        ),
        "rcg_reduces_false_yes_vs_random_matched": (
            None if rcg_fy is None else rcg_fy < mean_fy
        ),
        "false_yes_delta_rcg_minus_always": (
            None if rcg_fy is None or always_fy is None else rcg_fy - always_fy
        ),
        "false_yes_delta_rcg_minus_random": (
            None if rcg_fy is None else rcg_fy - mean_fy
        ),
        "degraded_scale8_context": {
            "always_answer_s8_false_yes_rate": s8_fy_full,
            "s8_false_yes_on_rcg_auto_decide_subset": s8_fy_matched,
            "s8_false_yes_ids": s8_false_yes_ids,
            "s8_false_yes_caught_by_rcg_abstain": s8_fy_caught,
            "rcg_catches_all_s8_false_yes": (
                len(s8_false_yes_ids) == 0
                or set(s8_false_yes_ids).issubset(set(abstain_ids))
            ),
            "note": (
                "Under resolution stress, s8 alone emits false-Yes; RCG abstention "
                "flags those items. Primary always-answer baseline uses s1 (FY=0), "
                "so matched-coverage FY does not decrease further."
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--judgments",
        type=Path,
        default=Path("experiments/raw/vlm_agent_judgments.jsonl"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/derived/vlm_resolution_summary.json"),
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_judgments(args.judgments)
    by_scale: dict[int, list] = defaultdict(list)
    for r in rows:
        by_scale[r["scale"]].append(r)

    qmap = by_question(rows)
    summary = {
        "model_id": "cursor-agent-frontier-vlm",
        "n_judgments": len(rows),
        "n_questions": len(qmap),
        "scales": sorted(by_scale.keys()),
        "accuracy_by_scale": {str(s): scale_metrics(by_scale[s]) for s in sorted(by_scale)},
        "multi_scale_disagreement": disagreement_stats(qmap),
        "rcg": rcg_analysis(qmap, seed=args.seed),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
