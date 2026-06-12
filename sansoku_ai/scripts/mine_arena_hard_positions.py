from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sansoku_ai.core import Move, Player, State, initial_state
from sansoku_ai.records import move_from_record, move_to_record, state_to_record


PHASES = (
    ("opening", 0, 8),
    ("early", 8, 16),
    ("mid", 16, 24),
    ("late", 24, 33),
)


@dataclass(frozen=True)
class MinedPosition:
    priority: float
    record: dict[str, Any]


def phase_for(moves_played: int) -> str:
    for name, lo, hi in PHASES:
        if lo <= moves_played < hi:
            return name
    return "unknown"


def state_key(state: State, move: Move, reason: str) -> str:
    payload = {
        "values": state.values,
        "owners": state.owners,
        "current": int(state.current),
        "first_score": state.first_score,
        "second_score": state.second_score,
        "moves_played": state.moves_played,
        "move": (move.row, move.col, move.value),
        "reason": reason,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def replay_moves(game: dict[str, Any]) -> list[tuple[State, dict[str, Any]]]:
    state = initial_state()
    states: list[tuple[State, dict[str, Any]]] = []
    for move_record in game.get("moves", []):
        states.append((state, move_record))
        state = state.apply(move_from_record(move_record))
    return states


def make_position_record(
    *,
    arena_name: str,
    serial: int,
    game: dict[str, Any],
    state: State,
    move_record: dict[str, Any],
    reasons: list[str],
    priority: float,
) -> dict[str, Any]:
    reason = ",".join(reasons)
    reason_slug = "_".join(reasons)
    move = move_from_record(move_record)
    policy = str(move_record.get("policy", "arena"))
    return {
        "id": f"{arena_name}_g{game['game']}_p{state.moves_played}_{reason_slug}_{serial}",
        "source_game": game["game"],
        "source_ply": state.moves_played,
        "phase": phase_for(state.moves_played),
        "state": state_to_record(state),
        "played_move": move_to_record(move),
        "played_policy": f"{policy}|mine:{reason}",
        "final_margin": int(game["first_score"]) - int(game["second_score"]),
        "candidate_side": int(game["candidate_side"]),
        "candidate_margin": int(game["candidate_margin"]),
        "raw_candidate_margin": int(game.get("raw_candidate_margin", 0)),
        "mine_reason": reason,
        "mine_priority": priority,
    }


def mine_game(
    *,
    arena_name: str,
    game: dict[str, Any],
    high_value: int,
    include_all_candidate_loss_moves: bool,
    include_opponent_turns: bool,
) -> list[MinedPosition]:
    if int(game.get("candidate_margin", 0)) >= 0:
        return []

    candidate_side = Player(int(game["candidate_side"]))
    states = replay_moves(game)
    mined: list[MinedPosition] = []
    serial = 0
    last_candidate: tuple[State, dict[str, Any]] | None = None

    def add(
        state: State,
        move_record: dict[str, Any],
        reasons: list[str],
        priority: float,
    ) -> None:
        nonlocal serial
        serial += 1
        if candidate_side == Player.SECOND:
            priority += 60.0
        record = make_position_record(
            arena_name=arena_name,
            serial=serial,
            game=game,
            state=state,
            move_record=move_record,
            reasons=reasons,
            priority=priority,
        )
        mined.append(MinedPosition(priority=priority, record=record))

    for state, move_record in states:
        actor = str(move_record.get("actor", ""))
        value = int(move_record["value"])
        if actor == "candidate":
            last_candidate = (state, move_record)
            if include_all_candidate_loss_moves:
                reasons = ["candidate_loss"]
                priority = 90.0
                if candidate_side == Player.SECOND:
                    reasons.append("candidate_second_loss")
                    priority += 90.0
                add(state, move_record, reasons, priority + value * 0.05)
            continue

        if actor == "opponent" and value >= high_value:
            if last_candidate is not None:
                prev_state, prev_move_record = last_candidate
                add(
                    prev_state,
                    prev_move_record,
                    ["allowed_opponent_high"],
                    260.0 + value,
                )
            if include_opponent_turns:
                add(
                    state,
                    move_record,
                    ["opponent_high_move"],
                    180.0 + value,
                )

    return mined


def summarize(records: list[dict[str, Any]]) -> str:
    reasons: dict[str, int] = {}
    current_sides: dict[int, int] = {1: 0, 2: 0}
    candidate_sides: dict[int, int] = {1: 0, 2: 0}
    weight_by_side: dict[int, float] = {1: 0.0, 2: 0.0}
    for record in records:
        for reason in str(record["mine_reason"]).split(","):
            reasons[reason] = reasons.get(reason, 0) + 1
        current = int(record["state"]["current"])
        candidate = int(record["candidate_side"])
        current_sides[current] = current_sides.get(current, 0) + 1
        candidate_sides[candidate] = candidate_sides.get(candidate, 0) + 1
        weight_by_side[current] = weight_by_side.get(current, 0.0) + float(
            record.get("mine_priority", 0.0)
        )
    reason_text = ",".join(f"{key}:{reasons[key]}" for key in sorted(reasons))
    return (
        f"reasons={reason_text} "
        f"current_side first={current_sides.get(1, 0)} second={current_sides.get(2, 0)} "
        f"candidate_side first={candidate_sides.get(1, 0)} second={candidate_sides.get(2, 0)} "
        f"priority_sum first={weight_by_side.get(1, 0.0):.1f} "
        f"second={weight_by_side.get(2, 0.0):.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arena", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--high-value", type=int, default=10)
    parser.add_argument("--id-prefix", default=None)
    parser.add_argument("--include-all-candidate-loss-moves", action="store_true")
    parser.add_argument("--include-opponent-turns", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.arena.read_text(encoding="utf-8"))
    games = payload.get("recorded_games") or []
    if not games:
        raise SystemExit(
            "arena has no recorded_games; rerun arena with --record-games "
            "--record-losses-only"
        )

    arena_name = args.id_prefix or args.arena.stem
    mined: list[MinedPosition] = []
    for game in games:
        mined.extend(
            mine_game(
                arena_name=arena_name,
                game=game,
                high_value=args.high_value,
                include_all_candidate_loss_moves=args.include_all_candidate_loss_moves,
                include_opponent_turns=args.include_opponent_turns,
            )
        )

    mined.sort(key=lambda item: item.priority, reverse=True)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in mined:
        move = move_from_record(item.record["played_move"])
        key = state_key(
            State(
                values=tuple(item.record["state"]["values"]),
                owners=tuple(item.record["state"]["owners"]),
                current=Player(int(item.record["state"]["current"])),
                first_score=int(item.record["state"]["first_score"]),
                second_score=int(item.record["state"]["second_score"]),
                moves_played=int(item.record["state"]["moves_played"]),
            ),
            move,
            str(item.record["mine_reason"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item.record)
        if len(deduped) >= args.limit:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in deduped:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    print(
        f"read_games={len(games)} mined={len(mined)} selected={len(deduped)} "
        f"high_value={args.high_value} wrote={args.output} {summarize(deduped)}"
    )


if __name__ == "__main__":
    main()
