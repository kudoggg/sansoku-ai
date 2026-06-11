from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from sansoku_ai.jsonl import iter_jsonl_records
from sansoku_ai.records import move_from_record, move_to_record, replay_game_states, state_to_record


PHASE_LIMITS = (
    ("opening", 0, 8),
    ("early", 8, 16),
    ("mid", 16, 24),
    ("late", 24, 32),
)


def phase_for(moves_played: int) -> str:
    for name, lo, hi in PHASE_LIMITS:
        if lo <= moves_played < hi:
            return name
    return "terminal"


def load_games(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl_records(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("games", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-ply", type=int, default=4)
    parser.add_argument("--min-remaining", type=int, default=5)
    parser.add_argument("--opening", type=int, default=1)
    parser.add_argument("--early", type=int, default=2)
    parser.add_argument("--mid", type=int, default=3)
    parser.add_argument("--late", type=int, default=2)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--id-prefix", default="")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    phase_counts = {
        "opening": args.opening,
        "early": args.early,
        "mid": args.mid,
        "late": args.late,
    }
    games = load_games(args.games)
    if args.max_games is not None:
        games = games[: args.max_games]

    selected: list[dict[str, Any]] = []
    for game in games:
        by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for state, played_move_record in replay_game_states(game):
            if state.moves_played < args.min_ply or state.remaining() < args.min_remaining:
                continue
            phase = phase_for(state.moves_played)
            if phase not in phase_counts:
                continue
            played_move = move_from_record(played_move_record)
            record_id = f"g{game['game']}_p{state.moves_played}"
            if args.id_prefix:
                record_id = f"{args.id_prefix}_{record_id}"
            by_phase[phase].append(
                {
                    "id": record_id,
                    "source_game": int(game["game"]),
                    "source_ply": state.moves_played,
                    "phase": phase,
                    "state": state_to_record(state),
                    "played_move": move_to_record(played_move),
                    "played_policy": str(played_move_record["policy"]),
                    "final_first_score": int(game["first_score"]),
                    "final_second_score": int(game["second_score"]),
                    "final_margin": int(game["margin"]),
                }
            )

        for phase, count in phase_counts.items():
            if count <= 0:
                continue
            pool = by_phase.get(phase, [])
            if not pool:
                continue
            selected.extend(rng.sample(pool, k=min(count, len(pool))))

    selected.sort(key=lambda rec: (rec["source_game"], rec["source_ply"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in selected:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    phase_totals = CounterLike()
    for record in selected:
        phase_totals.add(record["phase"])
    print(
        f"games={len(games)} positions={len(selected)} wrote={args.output} "
        f"phases={phase_totals.summary()}"
    )


class CounterLike:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def add(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    def summary(self) -> str:
        return ",".join(f"{key}:{self.counts[key]}" for key in sorted(self.counts))


if __name__ == "__main__":
    main()
