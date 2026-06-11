from __future__ import annotations

from sansoku_ai.tests.test_engine import (
    test_exact_search_matches_naive_minimax_on_small_endgames,
    test_initial_legal_moves_are_on_empty_cells_and_dominated,
)


def main() -> None:
    test_initial_legal_moves_are_on_empty_cells_and_dominated()
    test_exact_search_matches_naive_minimax_on_small_endgames()
    print("ok")


if __name__ == "__main__":
    main()
