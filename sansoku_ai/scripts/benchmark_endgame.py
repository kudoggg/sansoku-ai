from __future__ import annotations

import argparse
import random

from sansoku_ai.core import State, initial_state, legal_moves
from sansoku_ai.players import GreedyPlayer
from sansoku_ai.search import AlphaBetaSearch


def make_state(remaining: int, seed: int) -> State:
    rng = random.Random(seed)
    state = initial_state()
    greedy = GreedyPlayer()
    while state.remaining() > remaining:
        moves = legal_moves(state)
        if rng.random() < 0.75:
            move = greedy.choose(state)
        else:
            move = rng.choice(moves)
        state = state.apply(move)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remaining", type=int, default=8)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    for seed in range(args.samples):
        state = make_state(args.remaining, seed)
        search = AlphaBetaSearch(depth=1, endgame_exact_remaining=args.remaining)
        result = search.choose(state)
        print(
            f"seed={seed} remaining={state.remaining()} "
            f"legal={len(legal_moves(state))} value={result.value:+.1f} "
            f"nodes={result.nodes} time={result.elapsed_sec:.3f}s"
        )


if __name__ == "__main__":
    main()
