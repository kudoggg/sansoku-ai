from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sansoku_ai.nn_ranker import NnRanker, evaluate_nn_ranker_model
from sansoku_ai.ranker import load_jsonl


def choose_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-mode", choices=("best", "policy"), default="best")
    parser.add_argument("--max-sample-weight", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    ranker = NnRanker.load(args.model, device=device)
    records = load_jsonl(args.data)
    metrics = evaluate_nn_ranker_model(
        ranker.model,
        records,
        batch_size=args.batch_size,
        device=device,
        target_mode=args.target_mode,
        max_sample_weight=args.max_sample_weight,
    )
    print(
        f"records={len(records)} loss={metrics['loss']:.4f} "
        f"top1={metrics['top1']:.3f} avg_best_rank={metrics['avg_best_rank']:.2f} "
        f"device={device}"
    )


if __name__ == "__main__":
    main()
