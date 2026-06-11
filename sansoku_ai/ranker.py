from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import (
    BOARD_SIZE,
    CORNERS,
    INITIAL_OWNER,
    Move,
    Player,
    State,
    is_edge_index,
    rc_to_index,
)
from .jsonl import load_jsonl_records
from .records import move_to_record, state_to_record


DIRECTIONS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


FEATURE_NAMES = (
    "bias",
    "value_norm",
    "value_log",
    "value_sq_norm",
    "value_minus_context_max",
    "value_ratio_context_max",
    "is_context_max_value",
    "rank_by_value_norm",
    "same_cell_candidates_norm",
    "row_norm",
    "col_norm",
    "center_distance_norm",
    "corner",
    "edge",
    "value_x_corner",
    "value_x_edge",
    "high_zero_one",
    "current_is_second",
    "moves_played_norm",
    "remaining_norm",
    "first_score_norm",
    "second_score_norm",
    "current_margin_norm",
    "legal_count_norm",
    "analyzed_count_norm",
    "occupied_neighbors_norm",
    "first_neighbors_norm",
    "second_neighbors_norm",
    "initial_neighbors_norm",
    "empty_neighbors_norm",
    "ones_0",
    "ones_1",
    "ones_2",
    "ones_3",
    "ones_4",
    "ones_5",
    "ones_6",
    "ones_7",
    "ones_8",
    "ones_9",
)


def softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    best = max(scores)
    exps = [math.exp(score - best) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


def dot(xs: list[float], ys: list[float]) -> float:
    return sum(x * y for x, y in zip(xs, ys))


def clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def neighbor_counts(state: dict[str, Any], row: int, col: int) -> tuple[int, int, int, int, int]:
    owners = state["owners"]
    occupied = 0
    first = 0
    second = 0
    initial = 0
    empty = 0
    for dr, dc in DIRECTIONS_8:
        rr = row + dr
        cc = col + dc
        if not (0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE):
            continue
        owner = int(owners[rc_to_index(rr, cc)])
        if owner == 0:
            empty += 1
        else:
            occupied += 1
            if owner == int(Player.FIRST):
                first += 1
            elif owner == int(Player.SECOND):
                second += 1
            elif owner == INITIAL_OWNER:
                initial += 1
    return occupied, first, second, initial, empty


def context_for_record(record: dict[str, Any]) -> dict[str, Any]:
    moves = record["moves"]
    values = [float(item["move"]["value"]) for item in moves]
    max_value = max(values) if values else 1.0
    sorted_values = sorted(values, reverse=True)
    value_ranks: dict[tuple[int, int], int] = {}
    for idx, value in enumerate(sorted_values):
        value_ranks.setdefault((int(value), idx), idx)

    same_cell_counts: dict[int, int] = {}
    for item in moves:
        index = int(item["move"]["index"])
        same_cell_counts[index] = same_cell_counts.get(index, 0) + 1

    return {
        "max_value": max_value,
        "sorted_values": sorted_values,
        "same_cell_counts": same_cell_counts,
    }


def rank_by_value_norm(context: dict[str, Any], value: float, move_position: int) -> float:
    sorted_values = context["sorted_values"]
    rank = 0
    for idx, item_value in enumerate(sorted_values):
        if item_value == value:
            rank = idx
            break
    denom = max(1, len(sorted_values) - 1)
    return rank / denom


def features_for_candidate(
    record: dict[str, Any],
    candidate: dict[str, Any],
    context: dict[str, Any],
    move_position: int,
) -> list[float]:
    state = record["state"]
    move = candidate["move"]
    row = int(move["row"])
    col = int(move["col"])
    index = int(move["index"])
    value = float(move["value"])
    ones = int(move["ones"])
    current = int(state["current"])
    first_score = float(state["first_score"])
    second_score = float(state["second_score"])
    current_margin = first_score - second_score if current == int(Player.FIRST) else second_score - first_score
    max_value = max(1.0, float(context["max_value"]))
    corner = 1.0 if index in CORNERS else 0.0
    edge = 1.0 if is_edge_index(index) else 0.0
    occupied, first, second, initial, empty = neighbor_counts(state, row, col)
    same_cell_count = context["same_cell_counts"].get(index, 1)
    analyzed_count = max(1, int(record["analyzed_count"]))
    legal_count = max(1, int(record["legal_count"]))

    center_distance = (abs(row - 2.5) + abs(col - 2.5)) / 5.0
    value_norm = clamp(value / 81.0, 0.0, 2.0)
    output = [
        1.0,
        value_norm,
        math.log1p(value) / math.log(82.0),
        value_norm * value_norm,
        clamp((value - max_value) / 81.0, -2.0, 2.0),
        clamp(value / max_value, 0.0, 2.0),
        1.0 if value == max_value else 0.0,
        rank_by_value_norm(context, value, move_position),
        same_cell_count / max(1, analyzed_count),
        row / (BOARD_SIZE - 1),
        col / (BOARD_SIZE - 1),
        center_distance,
        corner,
        edge,
        value_norm * corner,
        value_norm * edge,
        1.0 if value >= 10 and ones in (0, 1) else 0.0,
        1.0 if current == int(Player.SECOND) else 0.0,
        float(state["moves_played"]) / 32.0,
        float(state["remaining"]) / 32.0,
        clamp(first_score / 400.0, 0.0, 2.0),
        clamp(second_score / 400.0, 0.0, 2.0),
        clamp(current_margin / 200.0, -2.0, 2.0),
        clamp(legal_count / 60.0, 0.0, 2.0),
        clamp(analyzed_count / 24.0, 0.0, 2.0),
        occupied / 8.0,
        first / 8.0,
        second / 8.0,
        initial / 8.0,
        empty / 8.0,
    ]
    output.extend(1.0 if ones == digit else 0.0 for digit in range(10))
    return output


def feature_matrix(record: dict[str, Any]) -> list[list[float]]:
    context = context_for_record(record)
    return [
        features_for_candidate(record, candidate, context, idx)
        for idx, candidate in enumerate(record["moves"])
    ]


def record_for_state_moves(state: State, moves: list[Move] | tuple[Move, ...]) -> dict[str, Any]:
    return {
        "state": state_to_record(state),
        "legal_count": len(moves),
        "analyzed_count": len(moves),
        "moves": [{"move": move_to_record(move)} for move in moves],
    }


def score_moves(
    model: "LinearRanker",
    state: State,
    moves: list[Move] | tuple[Move, ...],
) -> list[tuple[Move, float]]:
    record = record_for_state_moves(state, moves)
    scores = model.score_record(record)
    return list(zip(moves, scores))


@dataclass
class LinearRanker:
    weights: list[float]
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @classmethod
    def zeros(cls) -> "LinearRanker":
        return cls(weights=[0.0] * len(FEATURE_NAMES))

    def score_features(self, xs: list[float]) -> float:
        return dot(self.weights, xs)

    def score_record(self, record: dict[str, Any]) -> list[float]:
        return [self.score_features(xs) for xs in feature_matrix(record)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "linear_ranker",
            "feature_names": list(self.feature_names),
            "weights": self.weights,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LinearRanker":
        payload = json.loads(path.read_text(encoding="utf-8"))
        feature_names = tuple(payload["feature_names"])
        if feature_names != FEATURE_NAMES:
            raise ValueError("feature names do not match this code version")
        return cls(weights=[float(x) for x in payload["weights"]], feature_names=feature_names)


def target_policy(record: dict[str, Any]) -> list[float]:
    return [float(item["policy"]) for item in record["moves"]]


def best_index(record: dict[str, Any]) -> int:
    return int(record["best_index"])


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_jsonl_records(path)


def evaluate_ranker(model: LinearRanker, records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {"loss": 0.0, "top1": 0.0, "avg_best_rank": 0.0}

    total_loss = 0.0
    total_weight = 0.0
    top1 = 0
    best_rank_sum = 0.0

    for record in records:
        scores = model.score_record(record)
        probs = softmax(scores)
        target = target_policy(record)
        weight = float(record.get("sample_weight", 1.0))
        loss = -sum(t * math.log(max(p, 1e-12)) for t, p in zip(target, probs))
        total_loss += weight * loss
        total_weight += weight

        predicted = max(range(len(scores)), key=lambda idx: scores[idx])
        gold = best_index(record)
        if predicted == gold:
            top1 += 1
        score_order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        best_rank_sum += score_order.index(gold) + 1

    return {
        "loss": total_loss / max(1e-12, total_weight),
        "top1": top1 / len(records),
        "avg_best_rank": best_rank_sum / len(records),
    }
