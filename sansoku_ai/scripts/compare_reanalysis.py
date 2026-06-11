from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from sansoku_ai.jsonl import iter_jsonl_records
from sansoku_ai.scripts.select_hard_positions import same_move, value_for_move


def load_by_id(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl_records(path):
        records[str(record["id"])] = record
    return records


def top_gap(record: dict[str, Any]) -> float:
    values = sorted((float(item["value"]) for item in record["moves"]), reverse=True)
    if len(values) < 2:
        return 9999.0
    return values[0] - values[1]


def move_label(move: dict[str, Any] | None) -> str:
    if move is None:
        return "None"
    return f"({move['row']},{move['col']})={move['value']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path, help="earlier/lighter reanalysis JSONL")
    parser.add_argument("new", type=Path, help="later/heavier reanalysis JSONL")
    parser.add_argument("--d5-output", type=Path, default=None)
    parser.add_argument("--d5-limit", type=int, default=100)
    parser.add_argument("--bad-gap", type=float, default=4.0)
    parser.add_argument("--disagree-bonus", type=float, default=20.0)
    parser.add_argument("--close-gap", type=float, default=2.0)
    args = parser.parse_args()

    old_records = load_by_id(args.old)
    new_records = load_by_id(args.new)
    common_ids = sorted(set(old_records) & set(new_records))

    best_same = 0
    played_same_as_new_best = 0
    played_bad_new = 0
    old_new_best_disagree_and_played_bad = 0
    new_close = 0
    gaps: list[float] = []
    value_shifts: list[float] = []
    d5_candidates: list[tuple[float, dict[str, Any]]] = []

    for record_id in common_ids:
        old = old_records[record_id]
        new = new_records[record_id]
        old_best = old.get("best_move")
        new_best = new.get("best_move")
        played = new.get("played_move")

        if same_move(old_best, new_best):
            best_same += 1
        if same_move(played, new_best):
            played_same_as_new_best += 1

        new_best_value = float(new["best_value"])
        played_value = value_for_move(new, played) if played is not None else None
        if played_value is None:
            played_value = new_best_value - args.bad_gap
        gap = new_best_value - played_value
        gaps.append(gap)
        if gap >= args.bad_gap:
            played_bad_new += 1

        shift = new_best_value - float(old["best_value"])
        value_shifts.append(shift)

        gap_top = top_gap(new)
        if gap_top <= args.close_gap:
            new_close += 1

        if not same_move(old_best, new_best) and gap >= args.bad_gap:
            old_new_best_disagree_and_played_bad += 1

        priority = 0.0
        if not same_move(old_best, new_best):
            priority += args.disagree_bonus
        if gap >= args.bad_gap:
            priority += 100.0 + gap
        if gap_top <= args.close_gap:
            priority += 50.0 + (args.close_gap - gap_top)
        if priority > 0:
            enriched = dict(new)
            enriched["compare_priority"] = priority
            enriched["compare_old_best_move"] = old_best
            enriched["compare_old_best_value"] = old["best_value"]
            enriched["compare_new_gap_best_minus_played"] = gap
            enriched["compare_new_top_gap"] = gap_top
            d5_candidates.append((priority, enriched))

    total = len(common_ids)
    print(f"common={total}")
    if total == 0:
        return

    print(f"best_move_same={best_same}/{total} ({100.0 * best_same / total:.1f}%)")
    print(
        "played_is_new_best="
        f"{played_same_as_new_best}/{total} ({100.0 * played_same_as_new_best / total:.1f}%)"
    )
    print(
        f"played_bad_in_new_gap>={args.bad_gap:g}: "
        f"{played_bad_new}/{total} ({100.0 * played_bad_new / total:.1f}%)"
    )
    print(
        "old_new_best_disagree_and_played_bad="
        f"{old_new_best_disagree_and_played_bad}/{total} "
        f"({100.0 * old_new_best_disagree_and_played_bad / total:.1f}%)"
    )
    print(f"new_close_top_gap<={args.close_gap:g}: {new_close}/{total}")
    print(
        f"gap best-played avg={mean(gaps):+.2f} "
        f"min={min(gaps):+.2f} max={max(gaps):+.2f}"
    )
    print(
        f"best_value shift new-old avg={mean(value_shifts):+.2f} "
        f"min={min(value_shifts):+.2f} max={max(value_shifts):+.2f}"
    )

    d5_candidates.sort(key=lambda item: item[0], reverse=True)
    if args.d5_output is not None:
        selected = d5_candidates[: args.d5_limit]
        args.d5_output.parent.mkdir(parents=True, exist_ok=True)
        with args.d5_output.open("w", encoding="utf-8") as f:
            for _priority, record in selected:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(f"wrote_d5_candidates={len(selected)} path={args.d5_output}")

    print("\ntop candidates:")
    for priority, record in d5_candidates[:10]:
        print(
            f"  id={record['id']} priority={priority:.2f} "
            f"old={move_label(record['compare_old_best_move'])} "
            f"new={move_label(record['best_move'])} "
            f"played={move_label(record['played_move'])} "
            f"gap={record['compare_new_gap_best_minus_played']:+.2f} "
            f"top_gap={record['compare_new_top_gap']:+.2f}"
        )


if __name__ == "__main__":
    main()
