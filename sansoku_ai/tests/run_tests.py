from __future__ import annotations

from sansoku_ai.tests.test_engine import (
    test_endgame_exact_wrapper_overrides_base_policy,
    test_exact_search_matches_naive_minimax_on_small_endgames,
    test_initial_legal_moves_are_on_empty_cells_and_dominated,
    test_jsonl_loader_skips_bad_lines,
    test_pv_cycle_promotion_gate_requires_real_edge,
    test_pv_dataset_accepts_extra_reanalysis_sources,
    test_pv_selfplay_dataset_uses_mcts_visit_policy,
)


def main() -> None:
    test_initial_legal_moves_are_on_empty_cells_and_dominated()
    test_exact_search_matches_naive_minimax_on_small_endgames()
    test_endgame_exact_wrapper_overrides_base_policy()
    test_jsonl_loader_skips_bad_lines()
    test_pv_selfplay_dataset_uses_mcts_visit_policy()
    test_pv_dataset_accepts_extra_reanalysis_sources()
    test_pv_cycle_promotion_gate_requires_real_edge()
    print("ok")


if __name__ == "__main__":
    main()
