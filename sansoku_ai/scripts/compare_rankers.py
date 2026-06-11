from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from sansoku_ai.ranker import LinearRanker, best_index, load_jsonl


def predicted_index(model: LinearRanker, record: dict) -> int:
    scores = model.score_record(record)
    return max(range(len(scores)), key=lambda idx: scores[idx])


def best_rank(model: LinearRanker, record: dict) -> int:
    scores = model.score_record(record)
    gold = best_index(record)
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    return order.index(gold) + 1


class Bucket:
    def __init__(self) -> None:
        self.total = 0
        self.a_correct = 0
        self.b_correct = 0
        self.both_correct = 0
        self.a_only = 0
        self.b_only = 0
        self.both_wrong = 0
        self.a_rank_sum = 0
        self.b_rank_sum = 0

    def add(self, *, a_ok: bool, b_ok: bool, a_rank: int, b_rank: int) -> None:
        self.total += 1
        self.a_correct += int(a_ok)
        self.b_correct += int(b_ok)
        self.both_correct += int(a_ok and b_ok)
        self.a_only += int(a_ok and not b_ok)
        self.b_only += int(b_ok and not a_ok)
        self.both_wrong += int(not a_ok and not b_ok)
        self.a_rank_sum += a_rank
        self.b_rank_sum += b_rank

    def line(self, name: str, a_name: str, b_name: str) -> str:
        if self.total == 0:
            return f"{name}: total=0"
        return (
            f"{name}: total={self.total} "
            f"{a_name}_top1={self.a_correct / self.total:.3f} "
            f"{b_name}_top1={self.b_correct / self.total:.3f} "
            f"{a_name}_rank={self.a_rank_sum / self.total:.2f} "
            f"{b_name}_rank={self.b_rank_sum / self.total:.2f} "
            f"both={self.both_correct} {a_name}_only={self.a_only} "
            f"{b_name}_only={self.b_only} both_wrong={self.both_wrong}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_a", type=Path)
    parser.add_argument("model_b", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--name-a", default="a")
    parser.add_argument("--name-b", default="b")
    parser.add_argument(
        "--high-quality-tiers",
        default="d5_root16_move12,d6_root24_move12,ru_d5_root16_move12",
    )
    args = parser.parse_args()

    model_a = LinearRanker.load(args.model_a)
    model_b = LinearRanker.load(args.model_b)
    records = load_jsonl(args.data)
    high_quality = {item.strip() for item in args.high_quality_tiers.split(",") if item.strip()}

    buckets: dict[str, Bucket] = defaultdict(Bucket)

    for record in records:
        gold = best_index(record)
        pred_a = predicted_index(model_a, record)
        pred_b = predicted_index(model_b, record)
        a_ok = pred_a == gold
        b_ok = pred_b == gold
        a_rank = best_rank(model_a, record)
        b_rank = best_rank(model_b, record)
        tier = str(record.get("tier", "unknown"))
        phase = str(record.get("phase", "unknown"))

        buckets["all"].add(a_ok=a_ok, b_ok=b_ok, a_rank=a_rank, b_rank=b_rank)
        buckets[f"tier:{tier}"].add(a_ok=a_ok, b_ok=b_ok, a_rank=a_rank, b_rank=b_rank)
        buckets[f"phase:{phase}"].add(a_ok=a_ok, b_ok=b_ok, a_rank=a_rank, b_rank=b_rank)
        if tier in high_quality:
            buckets["high_quality"].add(a_ok=a_ok, b_ok=b_ok, a_rank=a_rank, b_rank=b_rank)

    print(f"records={len(records)} data={args.data}")
    print(buckets["all"].line("all", args.name_a, args.name_b))
    if "high_quality" in buckets:
        print(buckets["high_quality"].line("high_quality", args.name_a, args.name_b))

    print("\nby tier:")
    for key in sorted(key for key in buckets if key.startswith("tier:")):
        print(buckets[key].line(key, args.name_a, args.name_b))

    print("\nby phase:")
    for key in sorted(key for key in buckets if key.startswith("phase:")):
        print(buckets[key].line(key, args.name_a, args.name_b))


if __name__ == "__main__":
    main()
