from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from sansoku_ai.core import Move, initial_state, legal_moves
from sansoku_ai.players import AlphaBetaPlayer, RankerUnionPlayer, with_endgame_exact
from sansoku_ai.ranker import RankerModel
from sansoku_ai.ranker_loader import load_ranker_model


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


@dataclass(frozen=True)
class GameTask:
    game_idx: int
    seed: int
    random_opening_plies: int
    opening_top_k: int
    opening_temperature: float
    policy_mix: tuple[tuple[str, float], ...]
    ranker_model: str | None
    endgame: int
    move_limit: int | None


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


def rng_for_game(seed: int, game_idx: int) -> random.Random:
    return random.Random(seed + game_idx * 1_000_003)


def play_mixed_game(
    game_idx: int,
    *,
    rng: random.Random,
    random_opening_plies: int,
    opening_top_k: int,
    opening_temperature: float,
    policy_mix: list[tuple[str, float]],
    ranker: RankerModel | None,
    endgame: int,
    move_limit: int | None,
) -> GameRecord:
    state = initial_state()
    players = {
        "ab2": with_endgame_exact(
            AlphaBetaPlayer(depth=2, endgame_exact_remaining=endgame, move_limit=move_limit),
            endgame_exact_remaining=endgame,
        ),
        "ab3": with_endgame_exact(
            AlphaBetaPlayer(depth=3, endgame_exact_remaining=endgame, move_limit=move_limit),
            endgame_exact_remaining=endgame,
        ),
    }
    if any(name.startswith("ru") for name, _weight in policy_mix):
        if ranker is None:
            raise ValueError("--ranker-model is required when policy mix includes ru players")
        players["ru2"] = with_endgame_exact(
            RankerUnionPlayer(
                ranker=ranker,
                depth=2,
                endgame_exact_remaining=endgame,
                move_limit=move_limit,
            ),
            endgame_exact_remaining=endgame,
        )
        players["ru3"] = with_endgame_exact(
            RankerUnionPlayer(
                ranker=ranker,
                depth=3,
                endgame_exact_remaining=endgame,
                move_limit=move_limit,
            ),
            endgame_exact_remaining=endgame,
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


def play_mixed_game_task(task: GameTask) -> GameRecord:
    ranker = load_ranker_model(Path(task.ranker_model)) if task.ranker_model else None
    return play_mixed_game(
        task.game_idx,
        rng=rng_for_game(task.seed, task.game_idx),
        random_opening_plies=task.random_opening_plies,
        opening_top_k=task.opening_top_k,
        opening_temperature=task.opening_temperature,
        policy_mix=list(task.policy_mix),
        ranker=ranker,
        endgame=task.endgame,
        move_limit=task.move_limit,
    )


def game_to_payload(record: GameRecord | dict[str, Any]) -> dict[str, Any]:
    return asdict(record) if isinstance(record, GameRecord) else record


def write_game(dst, record: GameRecord | dict[str, Any]) -> None:
    dst.write(json.dumps(game_to_payload(record), separators=(",", ":")) + "\n")


def load_existing_games(path: Path, *, max_games: int) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}

    games: dict[int, dict[str, Any]] = {}
    needs_cleanup = False
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                needs_cleanup = True
                continue
            game_idx = int(record.get("game", -1))
            if not (0 <= game_idx < max_games):
                needs_cleanup = True
                continue
            if game_idx in games:
                needs_cleanup = True
            games[game_idx] = record

    if needs_cleanup:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as dst:
            for game_idx in sorted(games):
                write_game(dst, games[game_idx])
        tmp.replace(path)

    return games


def iter_generated_games(
    game_indices: list[int],
    *,
    seed: int,
    random_opening_plies: int,
    opening_top_k: int,
    opening_temperature: float,
    policy_mix: tuple[tuple[str, float], ...],
    ranker_model: Path | None,
    endgame: int,
    move_limit: int | None,
    workers: int,
    ranker: RankerModel | None,
):
    if workers <= 1:
        for game_idx in game_indices:
            yield play_mixed_game(
                game_idx,
                rng=rng_for_game(seed, game_idx),
                random_opening_plies=random_opening_plies,
                opening_top_k=opening_top_k,
                opening_temperature=opening_temperature,
                policy_mix=list(policy_mix),
                ranker=ranker,
                endgame=endgame,
                move_limit=move_limit,
            )
        return

    tasks = [
        GameTask(
            game_idx=game_idx,
            seed=seed,
            random_opening_plies=random_opening_plies,
            opening_top_k=opening_top_k,
            opening_temperature=opening_temperature,
            policy_mix=policy_mix,
            ranker_model=str(ranker_model) if ranker_model is not None else None,
            endgame=endgame,
            move_limit=move_limit,
        )
        for game_idx in game_indices
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(play_mixed_game_task, tasks, chunksize=1)


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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    policy_mix = tuple(args.policy_mix or [("ab2", args.ab2_prob), ("ab3", 1.0 - args.ab2_prob)])
    ranker = load_ranker_model(args.ranker_model) if args.ranker_model is not None else None
    start = perf_counter()
    existing: dict[int, dict[str, Any]] = {}
    if args.output is not None and args.resume:
        existing = load_existing_games(args.output, max_games=args.games)
        if existing:
            print(
                f"resume output={args.output} existing_games={len(existing)} "
                f"pending={args.games - len(existing)}"
            )

    game_indices = [game_idx for game_idx in range(args.games) if game_idx not in existing]
    done_games = len(existing)
    total_margin = sum(int(record["margin"]) for record in existing.values())
    records: list[GameRecord | dict[str, Any]] = list(existing.values()) if args.output is None else []

    dst = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.resume else "w"
        dst = args.output.open(mode, encoding="utf-8")

    try:
        for record in iter_generated_games(
            game_indices,
            seed=args.seed,
            random_opening_plies=args.random_opening_plies,
            opening_top_k=args.opening_top_k,
            opening_temperature=args.opening_temperature,
            policy_mix=policy_mix,
            ranker_model=args.ranker_model,
            endgame=args.endgame,
            move_limit=args.move_limit,
            workers=max(1, args.workers),
            ranker=ranker,
        ):
            done_games += 1
            total_margin += record.margin
            if dst is not None:
                write_game(dst, record)
                dst.flush()
            else:
                records.append(record)
            if args.progress_every and done_games % args.progress_every == 0:
                elapsed = perf_counter() - start
                print(
                    f"games={done_games} elapsed={elapsed:.2f}s "
                    f"games_per_sec={done_games / elapsed:.3f}"
                )
    finally:
        if dst is not None:
            dst.close()

    elapsed = perf_counter() - start
    avg_margin = total_margin / max(1, done_games)
    print(
        f"done games={done_games}/{args.games} elapsed={elapsed:.2f}s "
        f"games_per_sec={done_games / elapsed if elapsed else 0:.3f} "
        f"avg_margin={avg_margin:+.2f}"
    )

    if args.output is not None:
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
