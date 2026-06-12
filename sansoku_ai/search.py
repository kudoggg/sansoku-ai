from __future__ import annotations

from dataclasses import dataclass
from math import inf
from time import perf_counter

from .core import CORNERS, Move, Player, State, is_edge_index, legal_moves
from .ranker import RankerModel, score_moves


EXACT = "EXACT"
LOWER = "LOWER"
UPPER = "UPPER"


@dataclass(frozen=True)
class EvalWeights:
    max_legal: float = 0.42
    top3: float = 0.29
    zero_one: float = 0.32
    edge: float = 0.10
    corner: float = 0.34
    high_count: float = 0.17
    trap_second: float = 0.08
    trap_late_scale: float = 0.08


@dataclass
class TTEntry:
    depth: int
    value: float
    flag: str
    best_move: Move | None


@dataclass
class SearchResult:
    move: Move | None
    value: float
    nodes: int
    exact: bool
    elapsed_sec: float


@dataclass(frozen=True)
class RootMoveAnalysis:
    move: Move
    value: float


@dataclass
class RootAnalysis:
    moves: list[RootMoveAnalysis]
    best_move: Move | None
    best_value: float
    nodes: int
    exact: bool
    elapsed_sec: float


class AlphaBetaSearch:
    def __init__(
        self,
        *,
        depth: int = 4,
        endgame_exact_remaining: int = 8,
        weights: EvalWeights | None = None,
        move_limit: int | None = None,
        komi: int = 0,
    ) -> None:
        self.depth = depth
        self.endgame_exact_remaining = endgame_exact_remaining
        self.weights = weights or EvalWeights()
        self.move_limit = move_limit
        self.komi = komi
        self.tt: dict[tuple[tuple[int, ...], tuple[int, ...], int, int, bool], TTEntry] = {}
        self.nodes = 0

    def choose(self, state: State) -> SearchResult:
        self.nodes = 0
        start = perf_counter()
        exact = state.remaining() <= self.endgame_exact_remaining
        value, move = self._negamax(
            state,
            depth=self.depth,
            alpha=-inf,
            beta=inf,
            force_exact=exact,
        )
        return SearchResult(
            move=move,
            value=value,
            nodes=self.nodes,
            exact=exact,
            elapsed_sec=perf_counter() - start,
        )

    def analyze_root(self, state: State, *, root_limit: int | None = None) -> RootAnalysis:
        """Evaluate root candidates and return values from the side-to-move view."""

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

    def _negamax(
        self,
        state: State,
        *,
        depth: int,
        alpha: float,
        beta: float,
        force_exact: bool,
    ) -> tuple[float, Move | None]:
        self.nodes += 1

        moves = legal_moves(state)
        if state.remaining() <= 0 or not moves:
            return float(state.margin_for(state.current, self.komi)), None

        if force_exact or state.remaining() <= self.endgame_exact_remaining:
            depth = state.remaining()
            force_exact = True
        elif depth <= 0:
            return self.evaluate(state, moves), None

        original_alpha = alpha
        tt_key = (*state.key(), depth, force_exact)
        entry = self.tt.get(tt_key)
        if entry and entry.depth >= depth:
            if entry.flag == EXACT:
                return entry.value, entry.best_move
            if entry.flag == LOWER:
                alpha = max(alpha, entry.value)
            elif entry.flag == UPPER:
                beta = min(beta, entry.value)
            if alpha >= beta:
                return entry.value, entry.best_move

        best_value = -inf
        best_move: Move | None = None
        ordered = self.order_moves(state, moves)
        if self.move_limit is not None and not force_exact:
            ordered = ordered[: self.move_limit]

        for move in ordered:
            child = state.apply(move)
            child_value, _ = self._negamax(
                child,
                depth=depth - 1,
                alpha=-beta,
                beta=-alpha,
                force_exact=force_exact,
            )
            value = -child_value
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        if best_value <= original_alpha:
            flag = UPPER
        elif best_value >= beta:
            flag = LOWER
        else:
            flag = EXACT
        self.tt[tt_key] = TTEntry(depth, best_value, flag, best_move)
        return best_value, best_move

    def evaluate(self, state: State, moves: tuple[Move, ...] | None = None) -> float:
        moves = legal_moves(state) if moves is None else moves
        base = float(state.margin_for(state.current, self.komi))
        if not moves:
            return base

        values = sorted((mv.value for mv in moves), reverse=True)
        max_legal = values[0]
        top3 = sum(values[:3]) / min(3, len(values))
        zero_one = max(
            (mv.value for mv in moves if mv.value >= 10 and mv.value % 10 in (0, 1)),
            default=0,
        )
        edge = max((mv.value for mv in moves if is_edge_index(mv.index)), default=0)
        corner = max((mv.value for mv in moves if mv.index in CORNERS), default=0)
        high_count = sum(1 for mv in moves if mv.value >= 10)
        trap = self.sandwich_trap_value(state)

        w = self.weights
        tactical = (
            w.max_legal * max_legal
            + w.top3 * top3
            + w.zero_one * zero_one
            + w.edge * edge
            + w.corner * corner
            + w.high_count * high_count
        )

        second_sign = 1.0 if state.current == Player.SECOND else -1.0
        late = 1.0 - state.remaining() / 32.0
        trap_bonus = second_sign * (w.trap_second + w.trap_late_scale * late) * trap
        return base + tactical + trap_bonus

    def order_moves(self, state: State, moves: tuple[Move, ...]) -> list[Move]:
        def key(move: Move) -> tuple[int, int, int, int, int]:
            corner_bonus = 1 if move.index in CORNERS else 0
            edge_bonus = 1 if is_edge_index(move.index) else 0
            zero_one_bonus = 1 if move.value >= 10 and move.value % 10 in (0, 1) else 0
            return (
                move.value,
                zero_one_bonus,
                corner_bonus,
                edge_bonus,
                -move.index,
            )

        return sorted(moves, key=key, reverse=True)

    def sandwich_trap_value(self, state: State) -> float:
        # This is intentionally only an evaluation feature. It must not remove
        # moves because the "late empty-9-empty favors second" idea is strategic,
        # not a proven dominance rule.
        from .core import iter_empty_sandwiches

        total = 0.0
        for _left, _center, _right, value, ones in iter_empty_sandwiches(state):
            total += max(ones, value % 10)
            if ones >= 7:
                total += 2.0
        return total


@dataclass(frozen=True)
class RankerUnionConfig:
    value_moves: int = 16
    ranker_moves: int = 8
    defense_moves: int = 4
    max_root_moves: int = 24


class RankerUnionSearch(AlphaBetaSearch):
    def __init__(
        self,
        *,
        ranker: RankerModel,
        depth: int = 4,
        endgame_exact_remaining: int = 8,
        weights: EvalWeights | None = None,
        move_limit: int | None = None,
        komi: int = 0,
        union: RankerUnionConfig | None = None,
    ) -> None:
        super().__init__(
            depth=depth,
            endgame_exact_remaining=endgame_exact_remaining,
            weights=weights,
            move_limit=move_limit,
            komi=komi,
        )
        self.ranker = ranker
        self.union = union or RankerUnionConfig()

    def choose(self, state: State) -> SearchResult:
        analysis = self.analyze_root(state)
        return SearchResult(
            move=analysis.best_move,
            value=analysis.best_value,
            nodes=analysis.nodes,
            exact=analysis.exact,
            elapsed_sec=analysis.elapsed_sec,
        )

    def analyze_root(self, state: State, *, root_limit: int | None = None) -> RootAnalysis:
        self.nodes = 0
        start = perf_counter()
        moves = legal_moves(state)
        if not moves:
            value = float(state.margin_for(state.current, self.komi))
            return RootAnalysis([], None, value, self.nodes, True, perf_counter() - start)

        force_exact = state.remaining() <= self.endgame_exact_remaining
        if force_exact:
            root_moves = self.order_moves(state, moves)
            depth = state.remaining()
        else:
            root_moves = self.root_union_moves(state, moves)
            if root_limit is not None:
                root_moves = root_moves[:root_limit]
            depth = self.depth

        analyzed: list[RootMoveAnalysis] = []
        best_move: Move | None = None
        best_value = -inf
        for move in root_moves:
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

    def root_union_moves(self, state: State, moves: tuple[Move, ...]) -> list[Move]:
        config = self.union
        selected: list[Move] = []
        seen: set[Move] = set()

        def add(move_list: list[Move]) -> None:
            for move in move_list:
                if move not in seen:
                    selected.append(move)
                    seen.add(move)
                if len(selected) >= config.max_root_moves:
                    return

        value_ordered = self.order_moves(state, moves)
        add(value_ordered[: config.value_moves])

        ranked = sorted(
            score_moves(self.ranker, state, moves),
            key=lambda item: (item[1], item[0].value, -item[0].index),
            reverse=True,
        )
        add([move for move, _score in ranked[: config.ranker_moves]])

        defense = self.defense_ordered_moves(state, moves)
        add(defense[: config.defense_moves])

        return selected

    def defense_ordered_moves(self, state: State, moves: tuple[Move, ...]) -> list[Move]:
        scored: list[tuple[int, int, int, Move]] = []
        for move in moves:
            child = state.apply(move)
            replies = legal_moves(child)
            opponent_max = max((reply.value for reply in replies), default=0)
            scored.append((opponent_max, -move.value, move.index, move))
        scored.sort()
        return [move for _opp_max, _neg_value, _index, move in scored]
