from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sansoku_ai.jsonl import iter_jsonl_records


@dataclass(frozen=True)
class ScoredRecord:
    priority: float
    reason: str
    record: dict[str, Any]


def same_move(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if a is None or b is None:
        return False
    return (
        int(a["row"]) == int(b["row"])
        and int(a["col"]) == int(b["col"])
        and int(a["value"]) == int(b["value"])
    )


def value_for_move(record: dict[str, Any], move: dict[str, Any]) -> float | None:
    for item in record["moves"]:
        if same_move(item["move"], move):
            return float(item["value"])
    return None


def score_record(
    record: dict[str, Any],
    *,
    min_gap: float,
    big_move_value: int,
    big_blunder_gap: float,
    close_gap: float,
    min_remaining: int,
    max_remaining: int,
) -> ScoredRecord | None:
    state = record["state"]
    remaining = int(state["remaining"])
    if remaining < min_remaining or remaining > max_remaining:
        return None

    best_move = record.get("best_move")
    played_move = record.get("played_move")
    if best_move is None or played_move is None:
        return None

    best_value = float(record["best_value"])
    played_value = value_for_move(record, played_move)
    played_was_analyzed = played_value is not None
    if played_value is None:
        played_value = best_value - min_gap

    gap = best_value - played_value
    top_values = sorted((float(item["value"]) for item in record["moves"]), reverse=True)
    top_gap = top_values[0] - top_values[1] if len(top_values) >= 2 else 9999.0
    close = top_gap <= close_gap
    mismatch = not same_move(best_move, played_move)
    played_big = int(played_move["value"]) >= big_move_value
    big_blunder = played_big and gap >= big_blunder_gap

    reasons: list[str] = []
    priority = 0.0

    if mismatch and gap >= min_gap:
        reasons.append("played_vs_best_mismatch")
        priority += 100.0 + gap
    if big_blunder:
        reasons.append("big_move_failed")
        priority += 200.0 + gap + int(played_move["value"]) * 0.1
    if close:
        reasons.append("close_top_moves")
        priority += 50.0 + max(0.0, close_gap - top_gap)
    if not played_was_analyzed and mismatch:
        reasons.append("played_not_in_root_limit")
        priority += 25.0

    if not reasons:
        return None

    enriched = dict(record)
    enriched["hard_reason"] = ",".join(reasons)
    enriched["hard_priority"] = priority
    enriched["played_reanalyzed_value"] = played_value
    enriched["best_minus_played"] = gap
    enriched["top_gap"] = top_gap
    return ScoredRecord(priority=priority, reason=enriched["hard_reason"], record=enriched)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reanalyzed", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--min-gap", type=float, default=2.0)
    parser.add_argument("--big-move-value", type=int, default=10)
    parser.add_argument("--big-blunder-gap", type=float, default=4.0)
    parser.add_argument("--close-gap", type=float, default=1.5)
    parser.add_argument("--min-remaining", type=int, default=7)
    parser.add_argument("--max-remaining", type=int, default=28)
    args = parser.parse_args()

    selected: list[ScoredRecord] = []
    total = 0
    for record in iter_jsonl_records(args.reanalyzed):
        total += 1
        scored = score_record(
            record,
            min_gap=args.min_gap,
            big_move_value=args.big_move_value,
            big_blunder_gap=args.big_blunder_gap,
            close_gap=args.close_gap,
            min_remaining=args.min_remaining,
            max_remaining=args.max_remaining,
        )
        if scored is not None:
            selected.append(scored)

    selected.sort(key=lambda item: item.priority, reverse=True)
    selected = selected[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item.record, separators=(",", ":")) + "\n")

    reason_counts: dict[str, int] = {}
    for item in selected:
        for reason in item.reason.split(","):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    reasons = ",".join(f"{key}:{reason_counts[key]}" for key in sorted(reason_counts))
    print(
        f"read={total} selected={len(selected)} wrote={args.output} "
        f"reasons={reasons}"
    )


if __name__ == "__main__":
    main()
