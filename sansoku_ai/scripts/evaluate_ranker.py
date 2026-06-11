from __future__ import annotations

import argparse
from pathlib import Path

from sansoku_ai.ranker import LinearRanker, evaluate_ranker, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()

    model = LinearRanker.load(args.model)
    records = load_jsonl(args.data)
    metrics = evaluate_ranker(model, records)
    print(
        f"records={len(records)} loss={metrics['loss']:.4f} "
        f"top1={metrics['top1']:.3f} avg_best_rank={metrics['avg_best_rank']:.2f}"
    )


if __name__ == "__main__":
    main()
