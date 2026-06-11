from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from sansoku_ai.jsonl import iter_jsonl_records


def print_counter(title: str, counter: Counter[int], *, top: int) -> None:
    total = counter.total()
    print(f"\n{title} total={total}")
    if total == 0:
        return
    for value, count in counter.most_common(top):
        pct = 100.0 * count / total
        print(f"  {value:>3}: {count:>6} ({pct:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    value_counts: Counter[int] = Counter()
    ones_counts: Counter[int] = Counter()
    policy_value_counts: dict[str, Counter[int]] = defaultdict(Counter)
    policy_ones_counts: dict[str, Counter[int]] = defaultdict(Counter)
    player_value_counts: dict[int, Counter[int]] = defaultdict(Counter)
    game_count = 0
    move_count = 0
    margins: list[int] = []

    for game in iter_jsonl_records(args.path):
        game_count += 1
        margins.append(int(game["margin"]))
        for move in game["moves"]:
            value = int(move["value"])
            policy = str(move["policy"])
            player = int(move["player"])
            value_counts[value] += 1
            ones_counts[value % 10] += 1
            policy_value_counts[policy][value] += 1
            policy_ones_counts[policy][value % 10] += 1
            player_value_counts[player][value] += 1
            move_count += 1

    print(f"games={game_count} moves={move_count}")
    if margins:
        print(
            f"margin avg={sum(margins) / len(margins):+.2f} "
            f"min={min(margins):+d} max={max(margins):+d}"
        )

    print_counter("values", value_counts, top=args.top)
    print_counter("ones digits", ones_counts, top=10)

    for policy in sorted(policy_value_counts):
        print_counter(f"values by policy={policy}", policy_value_counts[policy], top=args.top)
        print_counter(f"ones by policy={policy}", policy_ones_counts[policy], top=10)

    for player in sorted(player_value_counts):
        print_counter(f"values by player={player}", player_value_counts[player], top=args.top)


if __name__ == "__main__":
    main()
