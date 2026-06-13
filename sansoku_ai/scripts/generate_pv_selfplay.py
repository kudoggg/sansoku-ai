from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from time import perf_counter

import torch

from sansoku_ai.core import Move, initial_state, legal_moves
from sansoku_ai.policy_value import PolicyValueModel
from sansoku_ai.pv_players import PolicyValueAlphaBetaPlayer, PuctPlayer
from sansoku_ai.records import move_to_record
from sansoku_ai.search import AlphaBetaSearch


def choose_device(text: str) -> str:
    if text == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return text


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


def move_policy_key(move: Move) -> str:
    return f"{move.row},{move.col},{move.value}"


def stat_policy_key(stat: dict) -> str:
    return f"{int(stat['row'])},{int(stat['col'])},{int(stat['value'])}"


def sample_from_visit_policy(
    moves: tuple[Move, ...],
    stats: list[dict],
    *,
    rng: random.Random,
    temperature: float,
) -> Move | None:
    if not stats:
        return None
    stats_by_key = {stat_policy_key(stat): stat for stat in stats}
    visits = [
        float((stats_by_key.get(move_policy_key(move)) or {}).get("visits", 0.0))
        for move in moves
    ]
    if sum(visits) <= 0.0:
        return None
    if temperature <= 0.0:
        return max(
            moves,
            key=lambda move: (
                (stats_by_key.get(move_policy_key(move)) or {}).get("visits", 0),
                move.value,
                -move.index,
            ),
        )

    inv_temperature = 1.0 / max(1e-6, temperature)
    weights = [max(0.0, value) ** inv_temperature for value in visits]
    total = sum(weights)
    if total <= 0.0:
        return None
    pick = rng.random() * total
    acc = 0.0
    for move, weight in zip(moves, weights):
        acc += weight
        if pick <= acc:
            return move
    return moves[-1]


def make_player(
    spec: str,
    *,
    model: PolicyValueModel,
    endgame: int,
    move_limit: int | None,
    komi: int,
    nn_value_weight: float,
    puct_simulations: int,
    cpuct: float,
    puct_batch_size: int,
    puct_leaf_depth: int,
    puct_leaf_weight: float,
    puct_leaf_move_limit: int | None,
    root_dirichlet_alpha: float,
    root_noise_fraction: float,
):
    if spec.startswith("pvab"):
        return PolicyValueAlphaBetaPlayer(
            policy_value=model,
            depth=int(spec[4:]),
            endgame_exact_remaining=endgame,
            move_limit=move_limit,
            komi=komi,
            nn_value_weight=nn_value_weight,
            name=spec,
        )
    if spec.startswith("puct"):
        simulations_text = spec[4:]
        simulations = int(simulations_text) if simulations_text else puct_simulations
        return PuctPlayer(
            policy_value=model,
            simulations=simulations,
            cpuct=cpuct,
            komi=komi,
            endgame_exact_remaining=endgame,
            root_dirichlet_alpha=root_dirichlet_alpha,
            root_noise_fraction=root_noise_fraction,
            batch_size=puct_batch_size,
            leaf_ab_depth=puct_leaf_depth,
            leaf_ab_weight=puct_leaf_weight,
            leaf_ab_move_limit=puct_leaf_move_limit,
            name=spec,
        )
    raise ValueError(f"unknown policy-value selfplay player: {spec}")


def play_game(
    game_idx: int,
    *,
    rng: random.Random,
    player,
    endgame: int,
    exact_search: AlphaBetaSearch,
    opening_plies: int,
    opening_top_k: int,
    opening_temperature: float,
    visit_sampling_plies: int,
    visit_temperature: float,
) -> dict:
    state = initial_state()
    moves_out: list[dict] = []
    if hasattr(player, "set_rng"):
        player.set_rng(rng)

    while state.remaining() > 0:
        moves = legal_moves(state)
        if not moves:
            break
        mcts_policy: list[dict] = []
        if state.moves_played < opening_plies:
            move = softmax_choice(
                moves,
                rng=rng,
                top_k=opening_top_k,
                temperature=opening_temperature,
            )
            policy = "opening_softmax"
        elif endgame > 0 and state.remaining() <= endgame:
            result = exact_search.choose(state)
            if result.move is None:
                break
            move = result.move
            policy = "exact_endgame"
        else:
            move = player.choose(state)
            policy = player.name
            mcts_policy = list(getattr(player, "last_root_stats", []))
            if state.moves_played < visit_sampling_plies:
                sampled = sample_from_visit_policy(
                    moves,
                    mcts_policy,
                    rng=rng,
                    temperature=visit_temperature,
                )
                if sampled is not None:
                    move = sampled

        record = {
            "ply": state.moves_played + 1,
            "player": int(state.current),
            **move_to_record(move),
            "policy": policy,
            "first_score_before": state.first_score,
            "second_score_before": state.second_score,
            "remaining_before": state.remaining(),
        }
        if mcts_policy:
            record["mcts_policy"] = mcts_policy
        moves_out.append(record)
        state = state.apply(move)

    return {
        "game": game_idx,
        "first_score": state.first_score,
        "second_score": state.second_score,
        "margin": state.first_score - state.second_score,
        "moves": moves_out,
    }


def load_existing_games(path: Path, *, max_games: int) -> dict[int, dict]:
    if not path.exists():
        return {}

    games: dict[int, dict] = {}
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
                dst.write(json.dumps(games[game_idx], separators=(",", ":")) + "\n")
        tmp.replace(path)

    return games


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--player", default="puct100")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--endgame", type=int, default=4)
    parser.add_argument("--move-limit", type=int, default=8)
    parser.add_argument("--full-search", action="store_true")
    parser.add_argument("--komi", type=int, default=16)
    parser.add_argument("--nn-value-weight", type=float, default=1.0)
    parser.add_argument("--puct-simulations", type=int, default=100)
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--puct-batch-size", type=int, default=1)
    parser.add_argument("--puct-leaf-depth", type=int, default=0)
    parser.add_argument("--puct-leaf-weight", type=float, default=0.0)
    parser.add_argument("--puct-leaf-move-limit", type=int, default=8)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-noise-fraction", type=float, default=0.25)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--opening-top-k", type=int, default=8)
    parser.add_argument("--opening-temperature", type=float, default=4.0)
    parser.add_argument("--visit-sampling-plies", type=int, default=12)
    parser.add_argument("--visit-temperature", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    model = PolicyValueModel.load(args.model, device=device)
    player = make_player(
        args.player,
        model=model,
        endgame=args.endgame,
        move_limit=None if args.full_search else args.move_limit,
        komi=args.komi,
        nn_value_weight=args.nn_value_weight,
        puct_simulations=args.puct_simulations,
        cpuct=args.cpuct,
        puct_batch_size=args.puct_batch_size,
        puct_leaf_depth=args.puct_leaf_depth,
        puct_leaf_weight=args.puct_leaf_weight,
        puct_leaf_move_limit=args.puct_leaf_move_limit,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_noise_fraction=args.root_noise_fraction,
    )
    exact_search = AlphaBetaSearch(
        depth=1,
        endgame_exact_remaining=args.endgame,
        move_limit=None,
        komi=args.komi,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_games(args.output, max_games=args.games) if args.resume else {}
    if existing:
        print(
            f"resume output={args.output} existing_games={len(existing)} "
            f"pending={args.games - len(existing)}",
            flush=True,
        )

    start = perf_counter()
    done_games = len(existing)
    total_margin = sum(int(record["margin"]) for record in existing.values())
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as f:
        for game_idx in range(args.games):
            if game_idx in existing:
                continue
            game_rng = rng_for_game(args.seed, game_idx)
            record = play_game(
                game_idx,
                rng=game_rng,
                player=player,
                endgame=args.endgame,
                exact_search=exact_search,
                opening_plies=args.opening_plies,
                opening_top_k=args.opening_top_k,
                opening_temperature=args.opening_temperature,
                visit_sampling_plies=args.visit_sampling_plies,
                visit_temperature=args.visit_temperature,
            )
            done_games += 1
            total_margin += int(record["margin"])
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            f.flush()
            if args.progress_every and (game_idx + 1) % args.progress_every == 0:
                elapsed = perf_counter() - start
                print(
                    f"games={done_games}/{args.games} elapsed={elapsed:.2f}s "
                    f"games_per_sec={done_games / max(1e-9, elapsed):.3f} "
                    f"avg_margin={total_margin / max(1, done_games):+.2f}",
                    flush=True,
                )

    elapsed = perf_counter() - start
    print(
        f"done games={done_games}/{args.games} elapsed={elapsed:.2f}s "
        f"games_per_sec={done_games / max(1e-9, elapsed):.3f} "
        f"avg_margin={total_margin / max(1, done_games):+.2f} wrote={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
