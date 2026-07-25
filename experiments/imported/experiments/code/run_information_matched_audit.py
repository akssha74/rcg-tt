#!/usr/bin/env python3
"""Information-matched audit of privileged and deployable reliability scores."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_primary_multiseed as base


ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "experiments/derived/greatness_iteration3"
AIDER_SPLITS = DERIVED / "aider_splits_dedup.json"
HURRICANE_SPLITS = DERIVED / "hurricane_splits_dedup.json"
SEEDS = base.SEEDS
RELATIVE_SCALES = (1, 2, 4, 8)
VIM_DIMS = (64, 128, 256, 384)
KNN_VALUES = (1, 5, 10, 20, 50)
GAMMAS = (0.5, 0.7, 0.9)
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 260724


@torch.no_grad()
def predict_full(model, dataset, device, batch_size: int = 64) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits, probabilities, labels, features = [], [], [], []
    model.eval()
    for images, target in loader:
        images = images.to(device)
        current_logits, current_features = base.logits_and_features(model, images)
        logits.append(current_logits.cpu().numpy())
        probabilities.append(torch.softmax(current_logits, dim=1).cpu().numpy())
        labels.append(target.numpy())
        features.append(current_features.cpu().numpy())
    return {
        "logits": np.concatenate(logits),
        "probabilities": np.concatenate(probabilities),
        "labels": np.concatenate(labels),
        "features": np.concatenate(features),
    }


def infer_scale(
    model,
    items,
    image_getter,
    label_getter,
    scale: int,
    device,
) -> dict:
    return predict_full(
        model,
        base.ImageItems(
            items, image_getter, label_getter, train=False, scale=scale
        ),
        device,
    )


def logsumexp(array: np.ndarray) -> np.ndarray:
    maximum = array.max(axis=1, keepdims=True)
    return maximum[:, 0] + np.log(np.exp(array - maximum).sum(axis=1))


def knn_kth_distance(
    train_features: np.ndarray, query_features: np.ndarray, k: int = 5
) -> np.ndarray:
    bank = torch.from_numpy(base.normalized(train_features))
    query = torch.from_numpy(base.normalized(query_features))
    distances = []
    for start in range(0, len(query), 256):
        similarity = query[start : start + 256] @ bank.T
        neighbours = torch.topk(similarity, k=k, dim=1, largest=True).values
        distances.append((1.0 - neighbours[:, -1]).numpy())
    return np.concatenate(distances)


def js_received(ladder: dict[int, dict]) -> np.ndarray:
    native = ladder[1]["probabilities"]
    return np.mean(
        [
            base.js_divergence(native, ladder[relative]["probabilities"])
            for relative in RELATIVE_SCALES[1:]
        ],
        axis=0,
    )


def js_fine_reference(ladder: dict[int, dict]) -> np.ndarray:
    """Anchor on the received s=8 prediction and compare three finer views."""
    received = ladder[8]["probabilities"]
    return np.mean(
        [
            base.js_divergence(received, ladder[scale]["probabilities"])
            for scale in (1, 2, 4)
        ],
        axis=0,
    )


def fit_vim(
    train: dict,
    validation_received: dict,
    validation_errors: np.ndarray,
    model,
) -> dict:
    linear = model.classifier[-1]
    weight = linear.weight.detach().cpu().double()
    bias = linear.bias.detach().cpu().double()
    origin_tensor = -(torch.linalg.pinv(weight) @ bias)
    centered = torch.from_numpy(train["features"]).double() - origin_tensor
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, order]
    maxlogit_mean = float(np.max(train["logits"], axis=1).mean())
    candidates = {}
    for dimension in VIM_DIMS:
        basis = eigenvectors[:, :dimension]
        train_projection = centered @ basis
        train_residual = torch.linalg.vector_norm(
            centered - train_projection @ basis.T, dim=1
        )
        alpha = maxlogit_mean / max(float(train_residual.mean().item()), 1e-12)
        validation_score = vim_score(
            validation_received["features"],
            validation_received["logits"],
            origin_tensor.numpy(),
            basis.numpy(),
            alpha,
        )
        candidates[str(dimension)] = {
            "dimension": dimension,
            "alpha": alpha,
            "validation_error_auroc": base.safe_auroc(
                validation_errors, validation_score
            ),
        }
    selected = max(
        candidates.values(),
        key=lambda row: (row["validation_error_auroc"], -row["dimension"]),
    )
    basis = eigenvectors[:, : selected["dimension"]].numpy()
    return {
        "origin": origin_tensor.numpy(),
        "basis": basis,
        "alpha": selected["alpha"],
        "dimension": selected["dimension"],
        "candidates": candidates,
    }


def vim_score(
    features: np.ndarray,
    logits: np.ndarray,
    origin: np.ndarray,
    basis: np.ndarray,
    alpha: float,
) -> np.ndarray:
    centered = torch.from_numpy(features).double() - torch.from_numpy(origin)
    basis_tensor = torch.from_numpy(basis)
    projection = centered @ basis_tensor
    residual = torch.linalg.vector_norm(
        centered - projection @ basis_tensor.T, dim=1
    ).numpy()
    virtual_logit = alpha * residual
    return 1.0 / (1.0 + np.exp(np.clip(logsumexp(logits) - virtual_logit, -60, 60)))


def error_metrics(errors: np.ndarray, score: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(errors, score)
    eligible = np.where(tpr >= 0.95)[0]
    return {
        "auroc": base.safe_auroc(errors, score),
        "auprc": float(average_precision_score(errors, score)),
        "fpr95": float(fpr[eligible[0]]) if len(eligible) else 1.0,
    }


def bootstrap_difference(
    errors: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    values = []
    probabilities = np.full(len(errors), 1.0 / len(errors))
    while len(values) < BOOTSTRAP_REPEATS:
        batch_size = min(250, BOOTSTRAP_REPEATS - len(values))
        counts = rng.multinomial(len(errors), probabilities, size=batch_size)
        auc_a, valid_a = weighted_auc_batch(errors, score_a, counts)
        auc_b, valid_b = weighted_auc_batch(errors, score_b, counts)
        valid = valid_a & valid_b
        values.extend((auc_a[valid] - auc_b[valid]).tolist())
    values = values[:BOOTSTRAP_REPEATS]
    return {
        "mean": float(np.mean(values)),
        "ci95": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
        "repeats": len(values),
    }


def weighted_auc_batch(
    labels: np.ndarray, score: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    sorted_labels = labels[order]
    sorted_counts = counts[:, order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_score) != 0) + 1]
    positive = sorted_counts * sorted_labels
    negative = sorted_counts * (1 - sorted_labels)
    positive_group = np.add.reduceat(positive, starts, axis=1)
    negative_group = np.add.reduceat(negative, starts, axis=1)
    negative_before = np.cumsum(negative_group, axis=1) - negative_group
    concordant = (
        positive_group * (negative_before + 0.5 * negative_group)
    ).sum(axis=1)
    positive_total = positive.sum(axis=1)
    negative_total = negative.sum(axis=1)
    valid = (positive_total > 0) & (negative_total > 0)
    auc = np.full(len(counts), np.nan)
    auc[valid] = concordant[valid] / (
        positive_total[valid] * negative_total[valid]
    )
    return auc, valid


def threshold_transfer(
    validation_scores: dict[str, np.ndarray],
    test_scores: dict[str, np.ndarray],
    errors: np.ndarray,
    false_critical: np.ndarray,
) -> dict:
    output = {}
    for gamma in GAMMAS:
        output[str(gamma)] = {}
        for name, validation_score in validation_scores.items():
            threshold = float(np.quantile(validation_score, gamma))
            keep = test_scores[name] <= threshold
            output[str(gamma)][name] = {
                "threshold": threshold,
                "coverage": float(keep.mean()),
                "absolute_coverage_error": float(abs(keep.mean() - gamma)),
                "selective_risk": float(errors[keep].mean()) if np.any(keep) else None,
                "false_critical_rate": (
                    float(false_critical[keep].mean()) if np.any(keep) else None
                ),
                "n": int(keep.sum()),
            }
    return output


def data_for(corpus: str):
    if corpus == "aider":
        return base.aider_data(AIDER_SPLITS)
    return base.hurricane_data(HURRICANE_SPLITS)


def evaluate_seed(corpus: str, seed: int, device) -> dict:
    (
        train_items,
        val_items,
        test_items,
        image_getter,
        label_getter,
        false_critical_function,
        _,
        num_labels,
    ) = data_for(corpus)
    state = torch.load(
        DERIVED / corpus / f"seed_{seed}/best.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = base.build_model(num_labels)
    model.load_state_dict(state["state_dict"])
    model.to(device)

    train = infer_scale(
        model, train_items, image_getter, label_getter, 1, device
    )
    validation = {
        scale: infer_scale(
            model, val_items, image_getter, label_getter, scale, device
        )
        for scale in base.SCALES
    }
    validation_received = {
        relative: infer_scale(
            model, val_items, image_getter, label_getter, relative, device
        )
        for relative in RELATIVE_SCALES
    }
    validation_prediction = validation[1]["probabilities"].argmax(1)
    validation_errors = (
        validation_prediction != validation[1]["labels"]
    ).astype(int)
    validation_s8_errors = (
        validation[8]["probabilities"].argmax(1) != validation[8]["labels"]
    ).astype(int)
    vim = fit_vim(train, validation[8], validation_s8_errors, model)
    knn_candidates = {
        str(k): base.safe_auroc(
            validation_s8_errors,
            knn_kth_distance(train["features"], validation[8]["features"], k),
        )
        for k in KNN_VALUES
    }
    selected_knn = max(
        KNN_VALUES, key=lambda k: (knn_candidates[str(k)], -k)
    )

    test_privileged = {
        scale: infer_scale(
            model, test_items, image_getter, label_getter, scale, device
        )
        for scale in base.SCALES
    }
    test_received = {
        relative: infer_scale(
            model,
            test_items,
            image_getter,
            label_getter,
            8 * relative,
            device,
        )
        for relative in RELATIVE_SCALES
    }
    received = test_received[1]
    prediction = received["probabilities"].argmax(1)
    errors = (prediction != received["labels"]).astype(int)
    false_critical = false_critical_function(received["labels"], prediction)

    train_features = train["features"]
    validation_native = validation_received[1]
    validation_uncertainty = {
        "confidence": 1.0 - validation_native["probabilities"].max(1),
        "maxlogit": -validation_native["logits"].max(1),
        "energy": -logsumexp(validation_native["logits"]),
        "eo_knn": knn_kth_distance(
            train_features, validation_native["features"], k=selected_knn
        ),
        "vim": vim_score(
            validation_native["features"],
            validation_native["logits"],
            vim["origin"],
            vim["basis"],
            vim["alpha"],
        ),
        "received_consistency": js_received(validation_received),
    }
    test_uncertainty = {
        "confidence": 1.0 - received["probabilities"].max(1),
        "maxlogit": -received["logits"].max(1),
        "energy": -logsumexp(received["logits"]),
        "eo_knn": knn_kth_distance(
            train_features, received["features"], k=selected_knn
        ),
        "vim": vim_score(
            received["features"],
            received["logits"],
            vim["origin"],
            vim["basis"],
            vim["alpha"],
        ),
        "received_consistency": js_received(test_received),
    }
    privileged_consistency = base.rcg_score(
        {
            scale: test_privileged[scale]["probabilities"]
            for scale in base.SCALES
        }
    )
    fine_reference_consistency = js_fine_reference(test_privileged)
    metrics = {
        name: error_metrics(errors, score)
        for name, score in test_uncertainty.items()
    }
    metrics["privileged_consistency"] = error_metrics(
        errors, privileged_consistency
    )
    metrics["fine_reference_consistency"] = error_metrics(
        errors, fine_reference_consistency
    )
    comparisons = {
        "privileged_minus_received": bootstrap_difference(
            errors,
            privileged_consistency,
            test_uncertainty["received_consistency"],
            BOOTSTRAP_SEED + seed,
        ),
        "fine_reference_minus_received": bootstrap_difference(
            errors,
            fine_reference_consistency,
            test_uncertainty["received_consistency"],
            BOOTSTRAP_SEED + seed + 1000,
        ),
    }
    for name, score in test_uncertainty.items():
        if name == "confidence":
            continue
        comparisons[f"{name}_minus_confidence"] = bootstrap_difference(
            errors,
            score,
            test_uncertainty["confidence"],
            BOOTSTRAP_SEED + seed + len(comparisons),
        )
    score_root = DERIVED / "information_scores"
    score_root.mkdir(parents=True, exist_ok=True)
    score_path = score_root / f"{corpus}_seed_{seed}.npz"
    np.savez_compressed(
        score_path,
        labels=received["labels"].astype(np.int64),
        predictions=prediction.astype(np.int64),
        errors=errors.astype(np.int8),
        confidence=test_uncertainty["confidence"],
        maxlogit=test_uncertainty["maxlogit"],
        energy=test_uncertainty["energy"],
        eo_knn=test_uncertainty["eo_knn"],
        vim=test_uncertainty["vim"],
        received_consistency=test_uncertainty["received_consistency"],
        privileged_consistency=privileged_consistency,
        fine_reference_consistency=fine_reference_consistency,
    )
    return {
        "seed": seed,
        "n": len(errors),
        "n_errors": int(errors.sum()),
        "accuracy": float((1 - errors).mean()),
        "vim": {
            "dimension": vim["dimension"],
            "alpha": vim["alpha"],
            "candidates": vim["candidates"],
        },
        "eo_knn": {
            "k": selected_knn,
            "validation_candidates": knn_candidates,
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "threshold_transfer": threshold_transfer(
            validation_uncertainty,
            test_uncertainty,
            errors,
            false_critical,
        ),
        "per_example_scores": str(score_path.relative_to(ROOT)),
    }


def aggregate(seed_results: dict[str, dict]) -> dict:
    score_names = list(next(iter(seed_results.values()))["metrics"])
    output = {"scores": {}}
    for score in score_names:
        values = np.asarray(
            [
                seed_results[str(seed)]["metrics"][score]["auroc"]
                for seed in SEEDS
            ]
        )
        output["scores"][score] = {
            "auroc_values": values.tolist(),
            "mean_auroc": float(values.mean()),
            "std_auroc": float(values.std(ddof=1)),
        }
    inflation = np.asarray(
        [
            seed_results[str(seed)]["comparisons"][
                "privileged_minus_received"
            ]["mean"]
            for seed in SEEDS
        ]
    )
    lower = [
        seed_results[str(seed)]["comparisons"]["privileged_minus_received"][
            "ci95"
        ][0]
        for seed in SEEDS
    ]
    output["privileged_inflation"] = {
        "values": inflation.tolist(),
        "mean": float(inflation.mean()),
        "std": float(inflation.std(ddof=1)),
        "all_seed_bootstrap_lower_positive": all(value > 0 for value in lower),
        "all_seed_differences_positive": bool(np.all(inflation > 0)),
    }
    fine_gap = np.asarray(
        [
            seed_results[str(seed)]["comparisons"][
                "fine_reference_minus_received"
            ]["mean"]
            for seed in SEEDS
        ]
    )
    fine_lower = [
        seed_results[str(seed)]["comparisons"][
            "fine_reference_minus_received"
        ]["ci95"][0]
        for seed in SEEDS
    ]
    output["anchor_matched_fine_reference_gap"] = {
        "values": fine_gap.tolist(),
        "mean": float(fine_gap.mean()),
        "std": float(fine_gap.std(ddof=1)),
        "all_seed_bootstrap_lower_positive": all(value > 0 for value in fine_lower),
        "all_seed_differences_positive": bool(np.all(fine_gap > 0)),
        "status": "post-hoc mechanism sensitivity prompted by independent review",
    }
    return output


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    summary = {
        "protocol": {
            "seeds": list(SEEDS),
            "received_scale": 8,
            "relative_scales": list(RELATIVE_SCALES),
            "vim_dimensions": list(VIM_DIMS),
            "knn_values": list(KNN_VALUES),
            "gammas": list(GAMMAS),
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "device": str(device),
        }
    }
    for corpus in ("aider", "hurricane"):
        seeds = {}
        for seed in SEEDS:
            seeds[str(seed)] = evaluate_seed(corpus, seed, device)
            print(
                "INFORMATION_AUDIT_SEED_COMPLETE",
                corpus,
                seed,
                json.dumps(seeds[str(seed)]["metrics"]),
                flush=True,
            )
        summary[corpus] = {
            "seeds": seeds,
            "aggregate": aggregate(seeds),
        }
    summary["a1_privileged_inflation_pass"] = all(
        summary[corpus]["aggregate"]["privileged_inflation"][
            "all_seed_bootstrap_lower_positive"
        ]
        and summary[corpus]["aggregate"]["privileged_inflation"][
            "all_seed_differences_positive"
        ]
        for corpus in ("aider", "hurricane")
    )
    summary["a2_matched_baseline_complete"] = True
    output = DERIVED / "information_matched_audit.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print("INFORMATION_MATCHED_AUDIT_COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
