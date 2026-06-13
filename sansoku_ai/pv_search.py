from __future__ import annotations

from math import inf
from time import perf_counter

from .core import Move, State, legal_moves
from .policy_value import PolicyValueModel
from .search import AlphaBetaSearch, RootAnalysis, RootMoveAnalysis, SearchResult


class PolicyValueAlphaBetaSearch(AlphaBetaSearch):
    """Alpha-beta search whose leaf evaluation is supplied by a policy-value net.

    This is the bridge stage between the hand-written evaluator and PUCT.  The
    network policy is used for move ordering; the value head replaces some or all
    of the hand-written leaf evaluation.
    """

    def __init__(
        self,
        *,
        policy_value: PolicyValueModel,
        depth: int = 3,
        endgame_exact_remaining: int = 4,
        move_limit: int | None = 8,
        komi: int = 0,
        nn_value_weight: float = 1.0,
    ) -> None:
        super().__init__(
            depth=depth,
            endgame_exact_remaining=endgame_exact_remaining,
            move_limit=move_limit,
            komi=komi,
        )
        self.policy_value = policy_value
        self.nn_value_weight = max(0.0, min(1.0, nn_value_weight))

    def choose(self, state: State) -> SearchResult:
        self.policy_value.clear_cache()
        return super().choose(state)

    def analyze_root(self, state: State, *, root_limit: int | None = None) -> RootAnalysis:
        self.policy_value.clear_cache()
        self.nodes = 0
        start = perf_counter()
        moves = legal_moves(state)
        if not moves:
            value = float(state.margin_for(state.current, self.komi))
            return RootAnalysis([], None, value, self.nodes, True, perf_counter() - start)

        force_exact = state.remaining() <= self.endgame_exact_remaining
        depth = state.remaining() if force_exact else self.depth
        ordered = self.order_moves(state, moves)
        if root_limit is not None and not force_exact:
            ordered = ordered[:root_limit]

        analyzed: list[RootMoveAnalysis] = []
        best_move: Move | None = None
        best_value = -inf

        for move in ordered:
            child = state.apply(move)
            child_value, _ = self._negamax(
                child,
                depth=depth - 1,
                alpha=-inf,
                beta=inf,
                force_exact=force_exact,
            )
            value = -child_value
            analyzed.append(RootMoveAnalysis(move=move, value=value))
            if value > best_value:
                best_value = value
                best_move = move

        return RootAnalysis(
            moves=analyzed,
            best_move=best_move,
            best_value=best_value,
            nodes=self.nodes,
            exact=force_exact,
            elapsed_sec=perf_counter() - start,
        )

    def evaluate(self, state: State, moves: tuple[Move, ...] | None = None) -> float:
        moves = legal_moves(state) if moves is None else moves
        if not moves:
            return float(state.margin_for(state.current, self.komi))

        classic = super().evaluate(state, moves)
        nn_raw = self.policy_value.value_raw(state, moves)
        komi_correction = state.margin_for(state.current, self.komi) - state.margin_for(state.current, 0)
        nn_value = nn_raw + komi_correction
        return (1.0 - self.nn_value_weight) * classic + self.nn_value_weight * nn_value

    def order_moves(self, state: State, moves: tuple[Move, ...]) -> list[Move]:
        if not moves:
            return []
        fallback = super().order_moves(state, moves)
        fallback_rank = {move: rank for rank, move in enumerate(fallback)}
        try:
            logits, _value = self.policy_value.predict_state(state, moves)
        except Exception:
            return fallback
        scored = []
        for move, logit in zip(moves, logits):
            # The tiny fallback term keeps ordering deterministic when the model
            # is uncertain and preserves the useful high-value-first alpha-beta behavior.
            fallback_bonus = -0.001 * fallback_rank.get(move, len(fallback))
            scored.append((float(logit) + fallback_bonus, move.value, -move.index, move))
        scored.sort(reverse=True)
        return [move for _score, _value, _idx, move in scored]
