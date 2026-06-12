from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from sansoku_ai.core import Player, State, initial_state, legal_moves, render_board
from sansoku_ai.players import (
    AlphaBetaPlayer,
    GreedyPlayer,
    RandomPlayer,
    RankerUnionPlayer,
    SansokuPlayer,
)
from sansoku_ai.ranker import RankerModel
from sansoku_ai.ranker_loader import load_ranker_model


@dataclass
class GameResult:
    first_score: int
    second_score: int
    moves: int

    @property
    def margin(self) -> int:
        return self.first_score - self.second_score


def make_player(
    spec: str,
    *,
    endgame_exact_remaining: int,
    move_limit: int | None,
    ranker: RankerModel | None,
    union_value_moves: int,
    union_ranker_moves: int,
    union_defense_moves: int,
    union_max_root_moves: int,
) -> SansokuPlayer:
    if spec == "greedy":
        return GreedyPlayer()
    if spec == "random":
        return RandomPlayer()
    if spec.startswith("ab"):
        depth = int(spec[2:])
        return AlphaBetaPlayer(
            depth=depth,
            endgame_exact_remaining=endgame_exact_remaining,
            move_limit=move_limit,
        )
    if spec.startswith("ru"):
        if ranker is None:
            raise ValueError("--ranker-model is required for ru players")
        depth = int(spec[2:])
        return RankerUnionPlayer(
            ranker=ranker,
            depth=depth,
            endgame_exact_remaining=endgame_exact_remaining,
            move_limit=move_limit,
            value_moves=union_value_moves,
            ranker_moves=union_ranker_moves,
            defense_moves=union_defense_moves,
            max_root_moves=union_max_root_moves,
        )
    raise ValueError(f"unknown player spec: {spec}")


def play_game(first: SansokuPlayer, second: SansokuPlayer, *, verbose: bool = False) -> GameResult:
    state = initial_state()
    players = {Player.FIRST: first, Player.SECOND: second}

    while state.remaining() > 0:
        moves = legal_moves(state)
        if not moves:
            break
        player = players[state.current]
        move = player.choose(state)
        if verbose:
            print(
                f"{state.moves_played + 1:02d}. {state.current.name} {player.name}: "
                f"({move.row},{move.col})={move.value}"
            )
        state = state.apply(move)

    if verbose:
        print(render_board(state))
    return GameResult(state.first_score, state.second_score, state.moves_played)


def candidate_margin(result: GameResult, *, candidate_side: Player, komi: int) -> int:
    first_margin_with_komi = result.margin + komi
    return first_margin_with_komi if candidate_side == Player.FIRST else -first_margin_with_komi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", default="ab2")
    parser.add_argument("--second", default="greedy")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--endgame", type=int, default=8)
    parser.add_argument("--move-limit", type=int, default=None)
    parser.add_argument("--ranker-model", type=Path, default=None)
    parser.add_argument("--union-value-moves", type=int, default=16)
    parser.add_argument("--union-ranker-moves", type=int, default=8)
    parser.add_argument("--union-defense-moves", type=int, default=4)
    parser.add_argument("--union-max-root-moves", type=int, default=24)
    parser.add_argument("--komi", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    ranker = load_ranker_model(args.ranker_model) if args.ranker_model is not None else None

    first_wins = 0
    second_wins = 0
    draws = 0
    total_margin = 0

    for game_idx in range(args.games):
        if game_idx % 2 == 0:
            first_player = make_player(
                args.first,
                endgame_exact_remaining=args.endgame,
                move_limit=args.move_limit,
                ranker=ranker,
                union_value_moves=args.union_value_moves,
                union_ranker_moves=args.union_ranker_moves,
                union_defense_moves=args.union_defense_moves,
                union_max_root_moves=args.union_max_root_moves,
            )
            second_player = make_player(
                args.second,
                endgame_exact_remaining=args.endgame,
                move_limit=args.move_limit,
                ranker=ranker,
                union_value_moves=args.union_value_moves,
                union_ranker_moves=args.union_ranker_moves,
                union_defense_moves=args.union_defense_moves,
                union_max_root_moves=args.union_max_root_moves,
            )
            label = f"{args.first} as FIRST vs {args.second} as SECOND"
            candidate_side = Player.FIRST
        else:
            first_player = make_player(
                args.second,
                endgame_exact_remaining=args.endgame,
                move_limit=args.move_limit,
                ranker=ranker,
                union_value_moves=args.union_value_moves,
                union_ranker_moves=args.union_ranker_moves,
                union_defense_moves=args.union_defense_moves,
                union_max_root_moves=args.union_max_root_moves,
            )
            second_player = make_player(
                args.first,
                endgame_exact_remaining=args.endgame,
                move_limit=args.move_limit,
                ranker=ranker,
                union_value_moves=args.union_value_moves,
                union_ranker_moves=args.union_ranker_moves,
                union_defense_moves=args.union_defense_moves,
                union_max_root_moves=args.union_max_root_moves,
            )
            label = f"{args.first} as SECOND vs {args.second} as FIRST"
            candidate_side = Player.SECOND

        result = play_game(first_player, second_player, verbose=args.verbose)
        raw_candidate_margin = result.margin if candidate_side == Player.FIRST else -result.margin
        adjusted_candidate_margin = candidate_margin(
            result,
            candidate_side=candidate_side,
            komi=args.komi,
        )
        total_margin += adjusted_candidate_margin
        if adjusted_candidate_margin > 0:
            first_wins += 1
        elif adjusted_candidate_margin < 0:
            second_wins += 1
        else:
            draws += 1
        print(
            f"game {game_idx + 1}: {label}: "
            f"score {result.first_score}-{result.second_score}, "
            f"candidate_margin={adjusted_candidate_margin:+d}, "
            f"raw_candidate_margin={raw_candidate_margin:+d}, "
            f"komi={args.komi}, moves={result.moves}"
        )

    print(
        f"summary candidate={args.first}: "
        f"wins={first_wins}, losses={second_wins}, draws={draws}, "
        f"avg_margin={total_margin / args.games:+.2f}, komi={args.komi}"
    )


if __name__ == "__main__":
    main()
