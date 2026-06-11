from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from sansoku_ai.ranker import (
    FEATURE_NAMES,
    LinearRanker,
    best_index,
    evaluate_ranker,
    feature_matrix,
    load_jsonl,
    softmax,
    target_policy,
)


def target_for_record(record: dict, mode: str) -> list[float]:
    if mode == "policy":
        return target_policy(record)
    if mode == "best":
        target = [0.0] * len(record["moves"])
        target[best_index(record)] = 1.0
        return target
    raise ValueError(f"unknown target mode: {mode}")


def train_epoch(
    model: LinearRanker,
    records: list[dict],
    *,
    rng: random.Random,
    lr: float,
    l2: float,
    max_sample_weight: float,
    target_mode: str,
) -> float:
    rng.shuffle(records)
    total_loss = 0.0
    total_weight = 0.0

    for record in records:
        xs = feature_matrix(record)
        scores = [model.score_features(item) for item in xs]
        probs = softmax(scores)
        target = target_for_record(record, target_mode)
        weight = min(float(record.get("sample_weight", 1.0)), max_sample_weight)
        loss = -sum(t * math.log(max(p, 1e-12)) for t, p in zip(target, probs))
        total_loss += weight * loss
        total_weight += weight

        grad = [l2 * w for w in model.weights]
        for prob, target_prob, feats in zip(probs, target, xs):
            scale = weight * (prob - target_prob)
            for idx, feat in enumerate(feats):
                grad[idx] += scale * feat

        for idx in range(len(model.weights)):
            model.weights[idx] -= lr * grad[idx]

    return total_loss / max(1e-12, total_weight)


def print_top_weights(model: LinearRanker, *, top: int = 12) -> None:
    pairs = sorted(
        zip(FEATURE_NAMES, model.weights),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    print("top_weights:")
    for name, weight in pairs[:top]:
        print(f"  {name:>24}: {weight:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("data/val.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("models/linear_ranker.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--max-sample-weight", type=float, default=5.0)
    parser.add_argument("--target-mode", choices=("policy", "best"), default="best")
    parser.add_argument("--select-metric", choices=("loss", "top1"), default="top1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_records = load_jsonl(args.train)
    val_records = load_jsonl(args.val)
    model = LinearRanker.zeros()

    print(
        f"train={len(train_records)} val={len(val_records)} "
        f"features={len(FEATURE_NAMES)} output={args.output}"
    )
    print(f"initial_val={evaluate_ranker(model, val_records)}")

    best_val = float("inf")
    best_top1 = -1.0
    best_weights = list(model.weights)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_records,
            rng=rng,
            lr=args.lr,
            l2=args.l2,
            max_sample_weight=args.max_sample_weight,
            target_mode=args.target_mode,
        )
        train_metrics = evaluate_ranker(model, train_records)
        val_metrics = evaluate_ranker(model, val_records)
        improved = (
            val_metrics["loss"] < best_val
            if args.select_metric == "loss"
            else val_metrics["top1"] > best_top1
        )
        if improved:
            best_val = val_metrics["loss"]
            best_top1 = val_metrics["top1"]
            best_weights = list(model.weights)
        print(
            f"epoch={epoch:02d} sgd_loss={train_loss:.4f} "
            f"train_loss={train_metrics['loss']:.4f} train_top1={train_metrics['top1']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1']:.3f} "
            f"val_best_rank={val_metrics['avg_best_rank']:.2f}"
        )

    model.weights = best_weights
    model.save(args.output)
    print(
        f"saved={args.output} best_val_loss={best_val:.4f} "
        f"best_val_top1={best_top1:.3f} select_metric={args.select_metric}"
    )
    print_top_weights(model)


if __name__ == "__main__":
    main()
