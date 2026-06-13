from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from sansoku_ai.core import Move, Player, State, initial_state, legal_moves
from sansoku_ai.players import (
    AlphaBetaPlayer,
    GreedyPlayer,
    RankerUnionPlayer,
    SansokuPlayer,
    with_endgame_exact,
)
from sansoku_ai.ranker import RankerModel
from sansoku_ai.ranker_loader import load_ranker_model
from sansoku_ai.scripts.generate_mixed_games import parse_policy_mix, softmax_choice


@dataclass(frozen=True)
class ArenaGame:
    game: int
    candidate_side: int
    first_score: int
    second_score: int
    candidate_margin: int
    raw_candidate_margin: int
    moves: int
    elapsed_sec: float


def empty_stats() -> dict[str, float | int]:
    return {
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "total_margin": 0,
        "avg_margin": 0.0,
    }


def update_stats(stats: dict[str, float | int], margin: int) -> None:
    stats["games"] = int(stats["games"]) + 1
    stats["total_margin"] = int(stats["total_margin"]) + margin
    if margin > 0:
        stats["wins"] = int(stats["wins"]) + 1
    elif margin < 0:
        stats["losses"] = int(stats["losses"]) + 1
    else:
        stats["draws"] = int(stats["draws"]) + 1
    stats["avg_margin"] = int(stats["total_margin"]) / max(1, int(stats["games"]))


def side_key(side: int) -> str:
    return "first" if side == int(Player.FIRST) else "second"


def summarize_by_side(results: list[ArenaGame]) -> dict[str, dict[str, float | int]]:
    summary = {"first": empty_stats(), "second": empty_stats()}
    for result in results:
        update_stats(summary[side_key(result.candidate_side)], result.candidate_margin)
    return summary


def format_stats(label: str, stats: dict[str, float | int]) -> str:
    return (
        f"{label}: games={int(stats['games'])} wins={int(stats['wins'])} "
        f"losses={int(stats['losses'])} draws={int(stats['draws'])} "
        f"avg_margin={float(stats['avg_margin']):+.2f}"
    )


def choose_policy_value_device(text: str) -> str:
    if text == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ModuleNotFoundError:
            return "cpu"
    return text


class Actor:
    def choose(self, state: State, rng: random.Random) -> tuple[Move, str]:
        raise NotImplementedError


class FixedActor(Actor):
    def __init__(self, player: SansokuPlayer, label: str) -> None:
        self.player = player
        self.label = label

    def choose(self, state: State, rng: random.Random) -> tuple[Move, str]:
        return self.player.choose(state), self.label


class MixedActor(Actor):
    def __init__(self, players: dict[str, SansokuPlayer], mix: list[tuple[str, float]]) -> None:
        self.players = players
        self.mix = mix

    def choose_policy(self, rng: random.Random) -> str:
        total = sum(weight for _name, weight in self.mix)
        pick = rng.random() * total
        acc = 0.0
        for name, weight in self.mix:
            acc += weight
            if pick <= acc:
                return name
        return self.mix[-1][0]

    def choose(self, state: State, rng: random.Random) -> tuple[Move, str]:
        label = self.choose_policy(rng)
        return self.players[label].choose(state), label


def make_player(
    spec: str,
    *,
    ranker: RankerModel | None,
    policy_value: Any | None,
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
    union_value_moves: int,
    union_ranker_moves: int,
    union_defense_moves: int,
    union_max_root_moves: int,
) -> SansokuPlayer:
    if spec == "greedy":
        return with_endgame_exact(
            GreedyPlayer(),
            endgame_exact_remaining=endgame,
        )
    if spec.startswith("ab"):
        return with_endgame_exact(
            AlphaBetaPlayer(
                depth=int(spec[2:]),
                endgame_exact_remaining=endgame,
                move_limit=move_limit,
            ),
            endgame_exact_remaining=endgame,
        )
    if spec.startswith("ru"):
        if ranker is None:
            raise ValueError("ranker model is required for ru players")
        return with_endgame_exact(
            RankerUnionPlayer(
                ranker=ranker,
                depth=int(spec[2:]),
                endgame_exact_remaining=endgame,
                move_limit=move_limit,
                value_moves=union_value_moves,
                ranker_moves=union_ranker_moves,
                defense_moves=union_defense_moves,
                max_root_moves=union_max_root_moves,
            ),
            endgame_exact_remaining=endgame,
        )
    if spec.startswith("pvab"):
        if policy_value is None:
            raise ValueError("policy-value model is required for pvab players")
        from sansoku_ai.pv_players import PolicyValueAlphaBetaPlayer

        return PolicyValueAlphaBetaPlayer(
            policy_value=policy_value,
            depth=int(spec[4:]),
            endgame_exact_remaining=endgame,
            move_limit=move_limit,
            komi=komi,
            nn_value_weight=nn_value_weight,
            name=spec,
        )
    if spec.startswith("puct"):
        if policy_value is None:
            raise ValueError("policy-value model is required for puct players")
        from sansoku_ai.pv_players import PuctPlayer

        simulations_text = spec[4:]
        simulations = int(simulations_text) if simulations_text else puct_simulations
        return with_endgame_exact(
            PuctPlayer(
                policy_value=policy_value,
                simulations=simulations,
                cpuct=cpuct,
                komi=komi,
                endgame_exact_remaining=endgame,
                batch_size=puct_batch_size,
                leaf_ab_depth=puct_leaf_depth,
                leaf_ab_weight=puct_leaf_weight,
                leaf_ab_move_limit=puct_leaf_move_limit,
                name=spec,
            ),
            endgame_exact_remaining=endgame,
        )
    raise ValueError(f"unknown player spec: {spec}")


def make_fixed_actor(
    spec: str,
    *,
    ranker: RankerModel | None,
    policy_value: Any | None,
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
    union_value_moves: int,
    union_ranker_moves: int,
    union_defense_moves: int,
    union_max_root_moves: int,
) -> FixedActor:
    return FixedActor(
        make_player(
            spec,
            ranker=ranker,
            policy_value=policy_value,
            endgame=endgame,
            move_limit=move_limit,
            komi=komi,
            nn_value_weight=nn_value_weight,
            puct_simulations=puct_simulations,
            cpuct=cpuct,
            puct_batch_size=puct_batch_size,
            puct_leaf_depth=puct_leaf_depth,
            puct_leaf_weight=puct_leaf_weight,
            puct_leaf_move_limit=puct_leaf_move_limit,
            union_value_moves=union_value_moves,
            union_ranker_moves=union_ranker_moves,
            union_defense_moves=union_defense_moves,
            union_max_root_moves=union_max_root_moves,
        ),
        spec,
    )


def make_mixed_actor(
    mix: list[tuple[str, float]],
    *,
    ranker: RankerModel | None,
    policy_value: Any | None,
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
    union_value_moves: int,
    union_ranker_moves: int,
    union_defense_moves: int,
    union_max_root_moves: int,
) -> MixedActor:
    specs = {name for name, _weight in mix}
    players = {
        spec: make_player(
            spec,
            ranker=ranker,
            policy_value=policy_value,
            endgame=endgame,
            move_limit=move_limit,
            komi=komi,
            nn_value_weight=nn_value_weight,
            puct_simulations=puct_simulations,
            cpuct=cpuct,
            puct_batch_size=puct_batch_size,
            puct_leaf_depth=puct_leaf_depth,
            puct_leaf_weight=puct_leaf_weight,
            puct_leaf_move_limit=puct_leaf_move_limit,
            union_value_moves=union_value_moves,
            union_ranker_moves=union_ranker_moves,
            union_defense_moves=union_defense_moves,
            union_max_root_moves=union_max_root_moves,
        )
        for spec in specs
    }
    return MixedActor(players, mix)


def apply_random_opening(
    state: State,
    *,
    rng: random.Random,
    plies: int,
    top_k: int,
    temperature: float,
) -> State:
    while state.moves_played < plies and state.remaining() > 0:
        moves = legal_moves(state)
        if not moves:
            break
        state = state.apply(
            softmax_choice(moves, rng=rng, top_k=top_k, temperature=temperature)
        )
    return state


def play_arena_game(
    game_idx: int,
    *,
    rng: random.Random,
    candidate_side: Player,
    candidate_actor: Actor,
    opponent_actor: Actor,
    komi: int,
    opening_plies: int,
    opening_top_k: int,
    opening_temperature: float,
) -> tuple[ArenaGame, dict[str, Any]]:
    start = perf_counter()
    state = initial_state()
    move_records: list[dict[str, Any]] = []
    while state.moves_played < opening_plies and state.remaining() > 0:
        moves = legal_moves(state)
        if not moves:
            break
        move = softmax_choice(
            moves,
            rng=rng,
            top_k=opening_top_k,
            temperature=opening_temperature,
        )
        move_records.append(
            {
                "ply": state.moves_played + 1,
                "player": int(state.current),
                "actor": "opening",
                "policy": "opening_softmax",
                "row": move.row,
                "col": move.col,
                "value": move.value,
                "first_score_before": state.first_score,
                "second_score_before": state.second_score,
                "remaining_before": state.remaining(),
            }
        )
        state = state.apply(move)

    while state.remaining() > 0:
        moves = legal_moves(state)
        if not moves:
            break
        actor = candidate_actor if state.current == candidate_side else opponent_actor
        move, _label = actor.choose(state, rng)
        move_records.append(
            {
                "ply": state.moves_played + 1,
                "player": int(state.current),
                "actor": "candidate" if state.current == candidate_side else "opponent",
                "policy": _label,
                "row": move.row,
                "col": move.col,
                "value": move.value,
                "first_score_before": state.first_score,
                "second_score_before": state.second_score,
                "remaining_before": state.remaining(),
            }
        )
        state = state.apply(move)

    result = ArenaGame(
        game=game_idx,
        candidate_side=int(candidate_side),
        first_score=state.first_score,
        second_score=state.second_score,
        candidate_margin=state.margin_for(candidate_side, komi),
        raw_candidate_margin=state.margin_for(candidate_side),
        moves=state.moves_played,
        elapsed_sec=perf_counter() - start,
    )
    game_record = {
        "game": game_idx,
        "candidate_side": int(candidate_side),
        "first_score": state.first_score,
        "second_score": state.second_score,
        "candidate_margin": result.candidate_margin,
        "raw_candidate_margin": result.raw_candidate_margin,
        "komi": komi,
        "moves": move_records,
    }
    return result, game_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="ru3")
    parser.add_argument("--opponent-mix", type=parse_policy_mix, default=parse_policy_mix("ab2:0.5,ab3:0.5"))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ranker-model", type=Path, default=None)
    parser.add_argument("--opponent-ranker-model", type=Path, default=None)
    parser.add_argument("--policy-value-model", type=Path, default=None)
    parser.add_argument("--opponent-policy-value-model", type=Path, default=None)
    parser.add_argument("--policy-value-device", default="auto")
    parser.add_argument("--endgame", type=int, default=4)
    parser.add_argument("--candidate-move-limit", type=int, default=8)
    parser.add_argument("--opponent-move-limit", type=int, default=8)
    parser.add_argument("--full-candidate", action="store_true")
    parser.add_argument("--full-opponent", action="store_true")
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--opening-top-k", type=int, default=8)
    parser.add_argument("--opening-temperature", type=float, default=4.0)
    parser.add_argument("--union-value-moves", type=int, default=16)
    parser.add_argument("--union-ranker-moves", type=int, default=8)
    parser.add_argument("--union-defense-moves", type=int, default=4)
    parser.add_argument("--union-max-root-moves", type=int, default=24)
    parser.add_argument("--nn-value-weight", type=float, default=1.0)
    parser.add_argument("--puct-simulations", type=int, default=100)
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--puct-batch-size", type=int, default=1)
    parser.add_argument("--puct-leaf-depth", type=int, default=0)
    parser.add_argument("--puct-leaf-weight", type=float, default=0.0)
    parser.add_argument("--puct-leaf-move-limit", type=int, default=8)
    parser.add_argument("--komi", type=int, default=16)
    parser.add_argument("--allow-odd-games", action="store_true")
    parser.add_argument(
        "--record-games",
        action="store_true",
        help="Store move histories in the output JSON for later hard-position mining.",
    )
    parser.add_argument(
        "--record-losses-only",
        action="store_true",
        help="When recording games, keep only candidate losses.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    if args.games % 2 != 0 and not args.allow_odd_games:
        raise SystemExit(
            "--games must be even so the candidate plays first and second equally; "
            "pass --allow-odd-games for a one-off smoke test."
        )

    candidate_ranker = load_ranker_model(args.ranker_model) if args.ranker_model else None
    opponent_ranker_path = args.opponent_ranker_model or args.ranker_model
    opponent_ranker = load_ranker_model(opponent_ranker_path) if opponent_ranker_path else None
    candidate_pv = None
    opponent_pv = None
    pv_device = choose_policy_value_device(args.policy_value_device)
    if args.policy_value_model:
        from sansoku_ai.policy_value import PolicyValueModel

        candidate_pv = PolicyValueModel.load(args.policy_value_model, device=pv_device)
    opponent_pv_path = args.opponent_policy_value_model or args.policy_value_model
    if opponent_pv_path:
        from sansoku_ai.policy_value import PolicyValueModel

        opponent_pv = PolicyValueModel.load(opponent_pv_path, device=pv_device)
    rng = random.Random(args.seed)
    candidate_move_limit = None if args.full_candidate else args.candidate_move_limit
    opponent_move_limit = None if args.full_opponent else args.opponent_move_limit

    results: list[ArenaGame] = []
    recorded_games: list[dict[str, Any]] = []
    start = perf_counter()
    wins = losses = draws = 0
    total_margin = 0

    for game_idx in range(args.games):
        candidate_side = Player.FIRST if game_idx % 2 == 0 else Player.SECOND
        candidate_actor = make_fixed_actor(
            args.candidate,
            ranker=candidate_ranker,
            policy_value=candidate_pv,
            endgame=args.endgame,
            move_limit=candidate_move_limit,
            komi=args.komi,
            nn_value_weight=args.nn_value_weight,
            puct_simulations=args.puct_simulations,
            cpuct=args.cpuct,
            puct_batch_size=args.puct_batch_size,
            puct_leaf_depth=args.puct_leaf_depth,
            puct_leaf_weight=args.puct_leaf_weight,
            puct_leaf_move_limit=args.puct_leaf_move_limit,
            union_value_moves=args.union_value_moves,
            union_ranker_moves=args.union_ranker_moves,
            union_defense_moves=args.union_defense_moves,
            union_max_root_moves=args.union_max_root_moves,
        )
        opponent_actor = make_mixed_actor(
            args.opponent_mix,
            ranker=opponent_ranker,
            policy_value=opponent_pv,
            endgame=args.endgame,
            move_limit=opponent_move_limit,
            komi=args.komi,
            nn_value_weight=args.nn_value_weight,
            puct_simulations=args.puct_simulations,
            cpuct=args.cpuct,
            puct_batch_size=args.puct_batch_size,
            puct_leaf_depth=args.puct_leaf_depth,
            puct_leaf_weight=args.puct_leaf_weight,
            puct_leaf_move_limit=args.puct_leaf_move_limit,
            union_value_moves=args.union_value_moves,
            union_ranker_moves=args.union_ranker_moves,
            union_defense_moves=args.union_defense_moves,
            union_max_root_moves=args.union_max_root_moves,
        )
        result, game_record = play_arena_game(
            game_idx,
            rng=rng,
            candidate_side=candidate_side,
            candidate_actor=candidate_actor,
            opponent_actor=opponent_actor,
            komi=args.komi,
            opening_plies=args.opening_plies,
            opening_top_k=args.opening_top_k,
            opening_temperature=args.opening_temperature,
        )
        results.append(result)
        if args.record_games and (
            not args.record_losses_only or result.candidate_margin < 0
        ):
            recorded_games.append(game_record)
        total_margin += result.candidate_margin
        if result.candidate_margin > 0:
            wins += 1
        elif result.candidate_margin < 0:
            losses += 1
        else:
            draws += 1
        if args.progress_every and (game_idx + 1) % args.progress_every == 0:
            elapsed = perf_counter() - start
            print(
                f"games={game_idx + 1} elapsed={elapsed:.2f}s "
                f"wins={wins} losses={losses} draws={draws} "
                f"avg_margin={total_margin / (game_idx + 1):+.2f}"
            )

    elapsed = perf_counter() - start
    by_side = summarize_by_side(results)
    print(
        f"summary candidate={args.candidate} opponent_mix={args.opponent_mix} "
        f"games={args.games} komi={args.komi} wins={wins} losses={losses} draws={draws} "
        f"avg_margin={total_margin / args.games:+.2f} elapsed={elapsed:.2f}s"
    )
    print(
        "by_side "
        + format_stats("candidate_first", by_side["first"])
        + " | "
        + format_stats("candidate_second", by_side["second"])
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "candidate": args.candidate,
            "opponent_mix": args.opponent_mix,
            "games": args.games,
            "komi": args.komi,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "avg_margin": total_margin / args.games,
            "elapsed_sec": elapsed,
            "by_side": by_side,
            "results": [asdict(result) for result in results],
            "recorded_games": recorded_games,
        }
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
