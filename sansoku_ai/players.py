from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from .core import Move, State, legal_moves
from .ranker import LinearRanker
from .search import AlphaBetaSearch, RankerUnionConfig, RankerUnionSearch, SearchResult


class SansokuPlayer(Protocol):
    name: str

    def choose(self, state: State) -> Move:
        ...


@dataclass
class RandomPlayer:
    seed: int | None = None
    name: str = "random"

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def choose(self, state: State) -> Move:
        moves = legal_moves(state)
        if not moves:
            raise ValueError("no legal moves")
        return self.rng.choice(moves)


@dataclass
class GreedyPlayer:
    name: str = "greedy"

    def choose(self, state: State) -> Move:
        moves = legal_moves(state)
        if not moves:
            raise ValueError("no legal moves")
        return max(moves, key=lambda mv: (mv.value, mv.index))


class AlphaBetaPlayer:
    def __init__(
        self,
        *,
        depth: int = 2,
        endgame_exact_remaining: int = 8,
        move_limit: int | None = None,
        name: str | None = None,
    ) -> None:
        self.search = AlphaBetaSearch(
            depth=depth,
            endgame_exact_remaining=endgame_exact_remaining,
            move_limit=move_limit,
        )
        self.name = name or f"alphabeta{depth}"
        self.last_result: SearchResult | None = None

    def choose(self, state: State) -> Move:
        result = self.search.choose(state)
        self.last_result = result
        if result.move is None:
            raise ValueError("no legal moves")
        return result.move


class RankerUnionPlayer:
    def __init__(
        self,
        *,
        ranker: LinearRanker,
        depth: int = 4,
        endgame_exact_remaining: int = 8,
        move_limit: int | None = None,
        value_moves: int = 16,
        ranker_moves: int = 8,
        defense_moves: int = 4,
        max_root_moves: int = 24,
        name: str | None = None,
    ) -> None:
        self.search = RankerUnionSearch(
            ranker=ranker,
            depth=depth,
            endgame_exact_remaining=endgame_exact_remaining,
            move_limit=move_limit,
            union=RankerUnionConfig(
                value_moves=value_moves,
                ranker_moves=ranker_moves,
                defense_moves=defense_moves,
                max_root_moves=max_root_moves,
            ),
        )
        self.name = name or f"ranker_union{depth}"
        self.last_result: SearchResult | None = None

    def choose(self, state: State) -> Move:
        result = self.search.choose(state)
        self.last_result = result
        if result.move is None:
            raise ValueError("no legal moves")
        return result.move
