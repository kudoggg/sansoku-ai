from __future__ import annotations

import json
from math import inf
from random import Random
from pathlib import Path
from tempfile import TemporaryDirectory

from sansoku_ai.core import EMPTY_OWNER, Move, State, initial_state, legal_moves
from sansoku_ai.jsonl import load_jsonl_records
from sansoku_ai.players import GreedyPlayer, with_endgame_exact
from sansoku_ai.records import move_to_record
from sansoku_ai.search import AlphaBetaSearch
from sansoku_ai.scripts.build_pv_training_dataset import (
    build_records_from_games,
    records_from_extra_sources,
)
from sansoku_ai.scripts.build_training_dataset import SourceSpec
from sansoku_ai.scripts.run_pv_cycle import should_promote


def assert_dominance(moves: tuple[Move, ...]) -> None:
    seen: dict[tuple[int, int], int] = {}
    for move in moves:
        key = (move.index, move.ones)
        assert move.value >= seen.get(key, -1)
        seen[key] = move.value


def naive_exact(state: State) -> float:
    moves = legal_moves(state)
    if state.remaining() <= 0 or not moves:
        return float(state.margin_for(state.current))
    best = -inf
    for move in moves:
        best = max(best, -naive_exact(state.apply(move)))
    return best


def random_state(target_remaining: int, seed: int) -> State:
    rng = Random(seed)
    state = initial_state()
    greedy = GreedyPlayer()
    while state.remaining() > target_remaining:
        moves = legal_moves(state)
        assert moves
        if rng.random() < 0.7:
            move = greedy.choose(state)
        else:
            move = rng.choice(moves)
        state = state.apply(move)
    return state


def test_initial_legal_moves_are_on_empty_cells_and_dominated() -> None:
    state = initial_state()
    moves = legal_moves(state)
    assert moves
    assert_dominance(moves)
    for move in moves:
        assert state.owners[move.index] == EMPTY_OWNER


def test_exact_search_matches_naive_minimax_on_small_endgames() -> None:
    for seed in range(5):
        state = random_state(target_remaining=4, seed=seed)
        search = AlphaBetaSearch(depth=1, endgame_exact_remaining=8)
        result = search.choose(state)
        assert result.exact
        assert result.value == naive_exact(state)


def test_endgame_exact_wrapper_overrides_base_policy() -> None:
    player = with_endgame_exact(GreedyPlayer(), endgame_exact_remaining=4)
    for seed in range(3):
        state = random_state(target_remaining=4, seed=seed + 20)
        exact_value = naive_exact(state)
        move = player.choose(state)
        assert -naive_exact(state.apply(move)) == exact_value


def test_jsonl_loader_skips_bad_lines() -> None:
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bad.jsonl"
        path.write_text('{"id":1}\n{"id":\n{"id":2}\n', encoding="utf-8")
        records = load_jsonl_records(path, warn=False)
    assert [record["id"] for record in records] == [1, 2]


def test_pv_selfplay_dataset_uses_mcts_visit_policy() -> None:
    state = initial_state()
    moves = legal_moves(state)
    played = moves[0]
    mcts_best = moves[-1]
    stats = []
    for move in moves:
        visits = 20 if move == mcts_best else 1
        stats.append(
            {
                **move_to_record(move),
                "visits": visits,
                "visit_policy": float(visits),
                "prior": 1.0 / len(moves),
                "q": 0.25 if move == mcts_best else -0.1,
            }
        )
    game = {
        "game": 0,
        "first_score": played.value,
        "second_score": 0,
        "margin": played.value,
        "moves": [
            {
                "ply": 1,
                "player": int(state.current),
                **move_to_record(played),
                "policy": "puct16",
                "mcts_policy": stats,
            }
        ],
    }

    records, build_stats = build_records_from_games(
        [game],
        tier="test_puct",
        quality=1,
        sample_weight=1.0,
        first_weight=1.0,
        second_weight=1.0,
        komi=16,
        target_komi=0,
        value_scale=80.0,
        include_non_mcts=False,
    )

    assert build_stats.records == 1
    assert len(records) == 1
    record = records[0]
    assert record["best_move"] == move_to_record(mcts_best)
    assert record["final_margin_target"] == played.value
    assert record["komi"] == 16
    assert record["target_komi"] == 0
    assert abs(sum(item["policy"] for item in record["moves"]) - 1.0) < 1e-9


def test_pv_dataset_accepts_extra_reanalysis_sources() -> None:
    state = initial_state()
    moves = legal_moves(state)
    played = moves[0]
    best = moves[-1]
    raw = {
        "id": "extra_0",
        "state": {
            "values": list(state.values),
            "owners": list(state.owners),
            "current": int(state.current),
            "first_score": state.first_score,
            "second_score": state.second_score,
            "moves_played": state.moves_played,
        },
        "phase": "opening",
        "played_move": move_to_record(played),
        "best_move": move_to_record(best),
        "best_value": 12.0,
        "final_margin": 5,
        "legal_count": len(moves),
        "depth": 5,
        "endgame": 4,
        "root_limit": 16,
        "move_limit": 12,
        "exact": False,
        "moves": [
            {"move": move_to_record(played), "value": 1.0},
            {"move": move_to_record(best), "value": 12.0},
        ],
    }
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "extra.jsonl"
        path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
        records, stats = records_from_extra_sources(
            [SourceSpec(path, "extra_d5", 7.0, 45)],
            policy_temperature=6.0,
            min_analyzed_moves=2,
        )

    assert len(records) == 1
    assert stats[0].read == 1
    assert stats[0].kept == 1
    assert stats[0].skipped == 0
    assert records[0]["tier"] == "extra_d5"
    assert records[0]["sample_weight"] == 7.0
    assert records[0]["best_move"] == move_to_record(best)


def test_pv_cycle_promotion_gate_requires_real_edge() -> None:
    payload = {
        "games": 10,
        "wins": 5,
        "losses": 5,
        "draws": 0,
        "avg_margin": 1.0,
        "by_side": {"second": {"avg_margin": 0.0}},
    }
    promoted, reasons = should_promote(
        payload,
        min_margin=0.0,
        min_win_rate=0.5,
        min_second_margin=None,
    )
    assert not promoted
    assert any(text.startswith("wins>5: False") for text in reasons)

    payload["wins"] = 6
    payload["losses"] = 4
    promoted, _reasons = should_promote(
        payload,
        min_margin=0.0,
        min_win_rate=0.5,
        min_second_margin=None,
    )
    assert promoted
