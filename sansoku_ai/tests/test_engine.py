from __future__ import annotations

from math import inf
from random import Random
from tempfile import TemporaryDirectory
from pathlib import Path

from sansoku_ai.core import EMPTY_OWNER, Move, State, initial_state, legal_moves
from sansoku_ai.jsonl import load_jsonl_records
from sansoku_ai.players import GreedyPlayer, with_endgame_exact
from sansoku_ai.search import AlphaBetaSearch


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
