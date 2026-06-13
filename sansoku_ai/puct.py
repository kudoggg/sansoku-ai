from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from time import perf_counter

from .core import Move, State, legal_moves
from .policy_value import PolicyValueModel, denormalize_value, normalize_margin
from .search import AlphaBetaSearch, SearchResult


@dataclass
class PuctNode:
    state: State
    prior: float = 1.0
    move: Move | None = None
    parent: "PuctNode | None" = None
    visits: int = 0
    value_sum: float = 0.0
    pending: bool = False
    children: dict[Move, "PuctNode"] = field(default_factory=dict)

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    @property
    def value_avg(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class PuctSearch:
    """PUCT search with optional batched neural-network leaf evaluation."""

    def __init__(
        self,
        *,
        policy_value: PolicyValueModel,
        simulations: int = 100,
        cpuct: float = 1.5,
        komi: int = 16,
        endgame_exact_remaining: int = 4,
        root_dirichlet_alpha: float = 0.0,
        root_noise_fraction: float = 0.0,
        batch_size: int = 1,
        rng: random.Random | None = None,
    ) -> None:
        self.policy_value = policy_value
        self.simulations = simulations
        self.cpuct = cpuct
        self.komi = komi
        self.endgame_exact_remaining = endgame_exact_remaining
        self.root_dirichlet_alpha = max(0.0, root_dirichlet_alpha)
        self.root_noise_fraction = max(0.0, min(1.0, root_noise_fraction))
        self.batch_size = max(1, batch_size)
        self.rng = rng or random.Random()
        self.exact_search = AlphaBetaSearch(
            depth=1,
            endgame_exact_remaining=endgame_exact_remaining,
            move_limit=None,
            komi=komi,
        )
        self.nodes = 0
        self.last_root_stats: list[dict[str, float | int]] = []

    def choose(self, state: State) -> SearchResult:
        start = perf_counter()
        self.nodes = 0
        self.last_root_stats = []
        self.policy_value.clear_cache()

        if self.endgame_exact_remaining > 0 and state.remaining() <= self.endgame_exact_remaining:
            result = self.exact_search.choose(state)
            return SearchResult(
                move=result.move,
                value=result.value,
                nodes=result.nodes,
                exact=True,
                elapsed_sec=perf_counter() - start,
            )

        root = PuctNode(state=state)
        self._expand_or_evaluate(root)
        self._apply_root_noise(root)

        if not root.children:
            return SearchResult(
                move=None,
                value=float(state.margin_for(state.current, self.komi)),
                nodes=0,
                exact=state.remaining() <= 0,
                elapsed_sec=perf_counter() - start,
            )

        if self.batch_size <= 1:
            for _sim in range(max(1, self.simulations)):
                node = root
                path = [node]
                while node.expanded:
                    node = self._select_child(node)
                    path.append(node)
                value = self._expand_or_evaluate(node)
                self._backup(path, value)
        else:
            self._run_batched_simulations(root, max(1, self.simulations))

        best_move, best_child = max(
            root.children.items(),
            key=lambda item: (item[1].visits, -item[0].index, item[0].value),
        )
        visit_total = sum(child.visits for child in root.children.values())
        self.last_root_stats = [
            {
                "row": move.row,
                "col": move.col,
                "value": move.value,
                "index": move.index,
                "ones": move.ones,
                "visits": child.visits,
                "visit_policy": child.visits / max(1, visit_total),
                "prior": child.prior,
                "q": -child.value_avg,
            }
            for move, child in sorted(
                root.children.items(),
                key=lambda item: item[1].visits,
                reverse=True,
            )
        ]
        # child value is from the child side-to-move perspective, so negate it
        # to report the root side's value.
        root_value_norm = -best_child.value_avg
        return SearchResult(
            move=best_move,
            value=denormalize_value(root_value_norm, self.policy_value.value_scale),
            nodes=self.nodes,
            exact=False,
            elapsed_sec=perf_counter() - start,
        )

    def _apply_root_noise(self, root: PuctNode) -> None:
        if (
            self.root_noise_fraction <= 0.0
            or self.root_dirichlet_alpha <= 0.0
            or not root.children
        ):
            return
        children = list(root.children.values())
        noise = [
            self.rng.gammavariate(self.root_dirichlet_alpha, 1.0)
            for _child in children
        ]
        total = sum(noise)
        if total <= 0.0:
            return
        noise = [value / total for value in noise]
        keep = 1.0 - self.root_noise_fraction
        for child, noise_value in zip(children, noise):
            child.prior = keep * child.prior + self.root_noise_fraction * noise_value

    def _select_child(self, node: PuctNode) -> PuctNode:
        parent_visits = max(1, node.visits)

        def score(child: PuctNode) -> float:
            q = -child.value_avg if child.visits else 0.0
            u = self.cpuct * child.prior * math.sqrt(parent_visits) / (1 + child.visits)
            return q + u

        children = [child for child in node.children.values() if not child.pending]
        if not children:
            children = list(node.children.values())
        return max(
            children,
            key=lambda child: (score(child), child.prior, child.move.value if child.move else 0),
        )

    def _expand_or_evaluate(self, node: PuctNode) -> float:
        value, moves = self._prepare_evaluation(node)
        if value is not None:
            return value

        assert moves is not None
        logits, value = self.policy_value.predict_state(node.state, moves)
        priors = self._priors_from_logits(logits, len(moves))
        value = self._add_komi_correction(node.state, value)
        self._expand_node(node, moves, priors)
        return value

    def _prepare_evaluation(self, node: PuctNode) -> tuple[float | None, tuple[Move, ...] | None]:
        self.nodes += 1
        state = node.state
        moves = legal_moves(state)
        if state.remaining() <= 0 or not moves:
            return (
                normalize_margin(
                    float(state.margin_for(state.current, self.komi)),
                    self.policy_value.value_scale,
                ),
                None,
            )

        if self.endgame_exact_remaining > 0 and state.remaining() <= self.endgame_exact_remaining:
            result = self.exact_search.choose(state)
            self.nodes += result.nodes
            return normalize_margin(result.value, self.policy_value.value_scale), None

        return None, moves

    def _expand_node(
        self,
        node: PuctNode,
        moves: tuple[Move, ...],
        priors: list[float],
    ) -> None:
        if not priors or len(priors) != len(moves):
            uniform = 1.0 / len(moves)
            priors = [uniform for _move in moves]
        for move, prior in zip(moves, priors):
            node.children[move] = PuctNode(
                state=node.state.apply(move),
                prior=max(1e-6, float(prior)),
                move=move,
                parent=node,
            )

    def _priors_from_logits(self, logits: list[float], count: int) -> list[float]:
        if count <= 0 or len(logits) != count:
            return []
        best = max(logits)
        exps = [math.exp(max(-80.0, min(80.0, score - best))) for score in logits]
        total = sum(exps)
        if total <= 0.0:
            return []
        return [value / total for value in exps]

    def _add_komi_correction(self, state: State, value: float) -> float:
        komi_correction = (
            state.margin_for(state.current, self.komi)
            - state.margin_for(state.current, 0)
        ) / self.policy_value.value_scale
        return max(-1.0, min(1.0, value + komi_correction))

    def _run_batched_simulations(self, root: PuctNode, simulations: int) -> None:
        pending: list[tuple[PuctNode, list[PuctNode], tuple[Move, ...]]] = []
        completed = 0
        while completed + len(pending) < simulations:
            leaf = self._select_leaf_for_batch(root)
            if leaf is None:
                if pending:
                    completed += self._flush_pending(pending)
                    pending = []
                    continue
                break

            node, path, value, moves = leaf
            if value is not None:
                self._backup(path, value)
                completed += 1
                continue

            assert moves is not None
            node.pending = True
            pending.append((node, path, moves))
            if len(pending) >= self.batch_size:
                completed += self._flush_pending(pending)
                pending = []

        if pending:
            completed += self._flush_pending(pending)

    def _select_leaf_for_batch(
        self,
        root: PuctNode,
    ) -> tuple[PuctNode, list[PuctNode], float | None, tuple[Move, ...] | None] | None:
        node = root
        path = [node]
        while node.expanded:
            node = self._select_child(node)
            if node.pending:
                return None
            path.append(node)

        value, moves = self._prepare_evaluation(node)
        return node, path, value, moves

    def _flush_pending(
        self,
        pending: list[tuple[PuctNode, list[PuctNode], tuple[Move, ...]]],
    ) -> int:
        if not pending:
            return 0
        predictions = self.policy_value.predict_states_batch(
            [(node.state, moves) for node, _path, moves in pending]
        )
        for (node, path, moves), (logits, value) in zip(pending, predictions):
            priors = self._priors_from_logits(logits, len(moves))
            value = self._add_komi_correction(node.state, value)
            self._expand_node(node, moves, priors)
            node.pending = False
            self._backup(path, value)
        return len(pending)

    def _backup(self, path: list[PuctNode], value: float) -> None:
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value
