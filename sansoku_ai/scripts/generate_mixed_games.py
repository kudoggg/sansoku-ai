from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from sansoku_ai.core import Move, State, initial_state, legal_moves
from sansoku_ai.players import AlphaBetaPlayer, RankerUnionPlayer
from sansoku_ai.ranker import LinearRanker


@dataclass(frozen=True)
class PlyRecord:
    ply: int
    player: int
    row: int
    col: int
    value: int
    policy: str
    first_score_before: int
    second_score_before: int
    remaining_before: int


@dataclass(frozen=True)
class GameRecord:
    game: int
    first_score: int
    second_score: int
    margin: int
    moves: list[PlyRecord]


def parse_policy_mix(text: str) -> list[tuple[str, float]]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("policy mix cannot be empty")
    result: list[tuple[str, float]] = []
    for part in parts:
        try:
            name, weight_text = part.split(":", 1)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "policy mix entries must look like ab2:0.5"
            ) from exc
        weight = float(weight_text)
        if weight < 0:
            raise argparse.ArgumentTypeError("policy weights must be non-negative")
        result.append((name, weight))
    total = sum(weight for _name, weight in result)
    if total <= 0:
        raise argparse.ArgumentTypeError("at least one policy weight must be positive")
    return result


def choose_policy(mix: list[tuple[str, float]], rng: random.Random) -> str:
    total = sum(weight for _name, weight in mix)
    pick = rng.random() * total
    acc = 0.0
    for name, weight in mix:
        acc += weight
        if pick <= acc:
            return name
    return mix[-1][0]


def softmax_choice(
    moves: tuple[Move, ...],
    *,
    rng: random.Random,
    top_k: int,
    temperature: float,
) -> Move:
    ordered = sorted(moves, key=lambda mv: (mv.value, mv.index), reverse=True)
    pool = ordered[: max(1, min(top_k, len(ordered)))]
    if temperature <= 0:
        return pool[0]

    best = max(mv.value for mv in pool)
    weights = [math.exp((mv.value - best) / temperature) for mv in pool]
    total = sum(weights)
    pick = rng.random() * total
    acc = 0.0
    for move, weight in zip(pool, weights):
        acc += weight
        if pick <= acc:
            return move
    return pool[-1]


def play_mixed_game(
    game_idx: int,
    *,
    rng: random.Random,
    random_opening_plies: int,
    opening_top_k: int,
    opening_temperature: float,
    policy_mix: list[tuple[str, float]],
    ranker: LinearRanker | None,
    endgame: int,
    move_limit: int | None,
) -> GameRecord:
    state = initial_state()
    players = {
        "ab2": AlphaBetaPlayer(depth=2, endgame_exact_remaining=endgame, move_limit=move_limit),
        "ab3": AlphaBetaPlayer(depth=3, endgame_exact_remaining=endgame, move_limit=move_limit),
    }
    if any(name.startswith("ru") for name, _weight in policy_mix):
        if ranker is None:
            raise ValueError("--ranker-model is required when policy mix includes ru players")
        players["ru2"] = RankerUnionPlayer(
            ranker=ranker,
            depth=2,
            endgame_exact_remaining=endgame,
            move_limit=move_limit,
        )
        players["ru3"] = RankerUnionPlayer(
            ranker=ranker,
            depth=3,
            endgame_exact_remaining=endgame,
            move_limit=move_limit,
        )
    records: list[PlyRecord] = []

    while state.remaining() > 0:
        moves = legal_moves(state)
        if not moves:
            break

        if state.moves_played < random_opening_plies:
            move = softmax_choice(
                moves,
                rng=rng,
                top_k=opening_top_k,
                temperature=opening_temperature,
            )
            policy = "opening_softmax"
        else:
            policy = choose_policy(policy_mix, rng)
            if policy not in players:
                raise ValueError(f"unknown policy in mix: {policy}")
            move = players[policy].choose(state)

        records.append(
            PlyRecord(
                ply=state.moves_played + 1,
                player=int(state.current),
                row=move.row,
                col=move.col,
                value=move.value,
                policy=policy,
                first_score_before=state.first_score,
                second_score_before=state.second_score,
                remaining_before=state.remaining(),
            )
        )
        state = state.apply(move)

    return GameRecord(
        game=game_idx,
        first_score=state.first_score,
        second_score=state.second_score,
        margin=state.first_score - state.second_score,
        moves=records,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-opening-plies", type=int, default=4)
    parser.add_argument("--opening-top-k", type=int, default=8)
    parser.add_argument("--opening-temperature", type=float, default=4.0)
    parser.add_argument("--ab2-prob", type=float, default=0.5)
    parser.add_argument(
        "--policy-mix",
        type=parse_policy_mix,
        default=None,
        help="Weighted policy mix after the opening, e.g. ab2:0.35,ab3:0.35,ru2:0.15,ru3:0.15",
    )
    parser.add_argument("--ranker-model", type=Path, default=None)
    parser.add_argument("--endgame", type=int, default=6)
    parser.add_argument("--move-limit", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    policy_mix = args.policy_mix or [("ab2", args.ab2_prob), ("ab3", 1.0 - args.ab2_prob)]
    ranker = LinearRanker.load(args.ranker_model) if args.ranker_model is not None else None
    start = perf_counter()
    records: list[GameRecord] = []

    for game_idx in range(args.games):
        record = play_mixed_game(
            game_idx,
            rng=rng,
            random_opening_plies=args.random_opening_plies,
            opening_top_k=args.opening_top_k,
            opening_temperature=args.opening_temperature,
            policy_mix=policy_mix,
            ranker=ranker,
            endgame=args.endgame,
            move_limit=args.move_limit,
        )
        records.append(record)
        if args.progress_every and (game_idx + 1) % args.progress_every == 0:
            elapsed = perf_counter() - start
            print(
                f"games={game_idx + 1} elapsed={elapsed:.2f}s "
                f"games_per_sec={(game_idx + 1) / elapsed:.3f}"
            )

    elapsed = perf_counter() - start
    margins = [record.margin for record in records]
    print(
        f"done games={args.games} elapsed={elapsed:.2f}s "
        f"games_per_sec={args.games / elapsed:.3f} "
        f"avg_margin={sum(margins) / len(margins):+.2f}"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
