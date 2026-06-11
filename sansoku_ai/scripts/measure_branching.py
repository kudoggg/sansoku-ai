from __future__ import annotations

from sansoku_ai.core import initial_state, legal_moves
from sansoku_ai.players import GreedyPlayer


def main() -> None:
    state = initial_state()
    greedy = GreedyPlayer()
    counts: list[int] = []
    while state.remaining() > 0:
        moves = legal_moves(state)
        if not moves:
            break
        counts.append(len(moves))
        state = state.apply(greedy.choose(state))

    print(counts)
    print(f"moves={len(counts)} max={max(counts)} avg={sum(counts) / len(counts):.2f}")


if __name__ == "__main__":
    main()
