from __future__ import annotations

import random
from dataclasses import dataclass

from .core import Move, State
from .policy_value import PolicyValueModel
from .puct import PuctSearch
from .pv_search import PolicyValueAlphaBetaSearch
from .search import SearchResult


@dataclass
class PolicyValueAlphaBetaPlayer:
    policy_value: PolicyValueModel
    depth: int = 3
    endgame_exact_remaining: int = 4
    move_limit: int | None = 8
    komi: int = 0
    nn_value_weight: float = 1.0
    name: str = "pvab3"

    def __post_init__(self) -> None:
        self.search = PolicyValueAlphaBetaSearch(
            policy_value=self.policy_value,
            depth=self.depth,
            endgame_exact_remaining=self.endgame_exact_remaining,
            move_limit=self.move_limit,
            komi=self.komi,
            nn_value_weight=self.nn_value_weight,
        )
        self.last_result: SearchResult | None = None

    def choose(self, state: State) -> Move:
        result = self.search.choose(state)
        self.last_result = result
        if result.move is None:
            raise ValueError("no legal moves")
        return result.move

    @property
    def last_root_stats(self) -> list[dict[str, float | int]]:
        return []


@dataclass
class PuctPlayer:
    policy_value: PolicyValueModel
    simulations: int = 100
    cpuct: float = 1.5
    komi: int = 16
    endgame_exact_remaining: int = 4
    root_dirichlet_alpha: float = 0.0
    root_noise_fraction: float = 0.0
    batch_size: int = 1
    leaf_ab_depth: int = 0
    leaf_ab_weight: float = 0.0
    leaf_ab_move_limit: int | None = 8
    rng: random.Random | None = None
    name: str = "puct100"

    def __post_init__(self) -> None:
        self.search = PuctSearch(
            policy_value=self.policy_value,
            simulations=self.simulations,
            cpuct=self.cpuct,
            komi=self.komi,
            endgame_exact_remaining=self.endgame_exact_remaining,
            root_dirichlet_alpha=self.root_dirichlet_alpha,
            root_noise_fraction=self.root_noise_fraction,
            batch_size=self.batch_size,
            leaf_ab_depth=self.leaf_ab_depth,
            leaf_ab_weight=self.leaf_ab_weight,
            leaf_ab_move_limit=self.leaf_ab_move_limit,
            rng=self.rng,
        )
        self.last_result: SearchResult | None = None

    def set_rng(self, rng: random.Random) -> None:
        self.search.rng = rng

    def choose(self, state: State) -> Move:
        result = self.search.choose(state)
        self.last_result = result
        if result.move is None:
            raise ValueError("no legal moves")
        return result.move

    @property
    def last_root_stats(self) -> list[dict[str, float | int]]:
        return self.search.last_root_stats
