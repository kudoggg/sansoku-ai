from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without torch
    raise ModuleNotFoundError(
        "PyTorch is required for policy-value models. Install torch before using "
        "train_policy_value/evaluate_policy_value, pvab, or puct players."
    ) from exc

from .core import BOARD_SIZE, Move, State, legal_moves
from .nn_ranker import BOARD_CHANNEL_NAMES, board_channels_for_state
from .ranker import FEATURE_NAMES, best_index, feature_matrix, record_for_state_moves, target_policy


@dataclass(frozen=True)
class PolicyValueConfig:
    board_channels: int = len(BOARD_CHANNEL_NAMES)
    move_features: int = len(FEATURE_NAMES)
    conv_channels: int = 48
    board_hidden: int = 96
    hidden: int = 160
    dropout: float = 0.05
    value_scale: float = 80.0
    target_komi: int = 0


def normalize_margin(margin: float, scale: float) -> float:
    return max(-1.0, min(1.0, margin / max(1e-6, scale)))


def denormalize_value(value: float, scale: float) -> float:
    return max(-1.0, min(1.0, value)) * scale


def value_target(record: dict[str, Any], *, mode: str, scale: float) -> float:
    if mode == "search":
        margin = float(record["search_margin_target"])
    elif mode == "final":
        margin = float(record["final_margin_target"])
    elif mode == "blend":
        margin = 0.75 * float(record["search_margin_target"]) + 0.25 * float(
            record["final_margin_target"]
        )
    else:
        raise ValueError(f"unknown value target mode: {mode}")
    return normalize_margin(margin, scale)


class PolicyValueNet(nn.Module):
    def __init__(self, config: PolicyValueConfig) -> None:
        super().__init__()
        self.config = config
        self.board_net = nn.Sequential(
            nn.Conv2d(config.board_channels, config.conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.conv_channels, config.conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.conv_channels, config.conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(config.conv_channels * BOARD_SIZE * BOARD_SIZE, config.board_hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(config.board_hidden + config.move_features, config.hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(config.board_hidden, config.hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.hidden // 2),
            nn.ReLU(),
            nn.Linear(config.hidden // 2, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        boards: torch.Tensor,
        move_features: torch.Tensor,
        move_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        board_embedding = self.board_net(boards)
        batch_size, max_moves, _features = move_features.shape
        expanded = board_embedding[:, None, :].expand(batch_size, max_moves, -1)
        policy_logits = self.policy_head(torch.cat([expanded, move_features], dim=-1)).squeeze(-1)
        if move_mask is not None:
            policy_logits = policy_logits.masked_fill(~move_mask, -1e9)
        value = self.value_head(board_embedding).squeeze(-1)
        return policy_logits, value


def collate_policy_value_records(
    records: list[dict[str, Any]],
    *,
    value_target_mode: str = "search",
    value_scale: float = 80.0,
    max_sample_weight: float = 5.0,
) -> dict[str, torch.Tensor]:
    batch_size = len(records)
    max_moves = max(len(record["moves"]) for record in records)
    boards = torch.zeros(
        batch_size,
        len(BOARD_CHANNEL_NAMES),
        BOARD_SIZE,
        BOARD_SIZE,
        dtype=torch.float32,
    )
    moves = torch.zeros(batch_size, max_moves, len(FEATURE_NAMES), dtype=torch.float32)
    mask = torch.zeros(batch_size, max_moves, dtype=torch.bool)
    policy = torch.zeros(batch_size, max_moves, dtype=torch.float32)
    best = torch.zeros(batch_size, dtype=torch.long)
    values = torch.zeros(batch_size, dtype=torch.float32)
    weights = torch.ones(batch_size, dtype=torch.float32)

    for batch_idx, record in enumerate(records):
        boards[batch_idx] = torch.tensor(
            board_channels_for_state(record["state"]),
            dtype=torch.float32,
        )
        xs = feature_matrix(record)
        count = len(xs)
        moves[batch_idx, :count] = torch.tensor(xs, dtype=torch.float32)
        mask[batch_idx, :count] = True
        policy[batch_idx, :count] = torch.tensor(target_policy(record), dtype=torch.float32)
        best[batch_idx] = best_index(record)
        values[batch_idx] = value_target(
            record,
            mode=value_target_mode,
            scale=value_scale,
        )
        weights[batch_idx] = min(float(record.get("sample_weight", 1.0)), max_sample_weight)

    return {
        "boards": boards,
        "move_features": moves,
        "move_mask": mask,
        "policy": policy,
        "best": best,
        "value": values,
        "weights": weights,
    }


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def policy_loss(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    target_mode: str,
) -> torch.Tensor:
    weights = batch["weights"]
    if target_mode == "best":
        losses = F.cross_entropy(logits, batch["best"], reduction="none")
    elif target_mode == "policy":
        log_probs = F.log_softmax(logits, dim=1)
        losses = -(batch["policy"] * log_probs).sum(dim=1)
    else:
        raise ValueError(f"unknown policy target mode: {target_mode}")
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


def value_loss(values: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    weights = batch["weights"]
    losses = F.smooth_l1_loss(values, batch["value"], reduction="none")
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


def policy_value_loss(
    logits: torch.Tensor,
    values: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    policy_target_mode: str,
    value_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p_loss = policy_loss(logits, batch, target_mode=policy_target_mode)
    v_loss = value_loss(values, batch)
    return p_loss + value_weight * v_loss, p_loss, v_loss


@torch.no_grad()
def evaluate_policy_value_model(
    model: PolicyValueNet,
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
    policy_target_mode: str = "best",
    value_target_mode: str = "search",
    value_weight: float = 1.0,
    max_sample_weight: float = 5.0,
) -> dict[str, float]:
    if not records:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "top1": 0.0,
            "avg_best_rank": 0.0,
            "value_mae": 0.0,
            "value_sign_acc": 0.0,
        }

    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_weight = 0.0
    top1 = 0
    best_rank_sum = 0.0
    value_abs_error = 0.0
    value_sign_correct = 0
    config = model.config

    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        batch = move_batch_to_device(
            collate_policy_value_records(
                chunk,
                value_target_mode=value_target_mode,
                value_scale=config.value_scale,
                max_sample_weight=max_sample_weight,
            ),
            device,
        )
        logits, values = model(batch["boards"], batch["move_features"], batch["move_mask"])
        loss, p_loss, v_loss = policy_value_loss(
            logits,
            values,
            batch,
            policy_target_mode=policy_target_mode,
            value_weight=value_weight,
        )
        weight_sum = float(batch["weights"].sum().item())
        total_loss += float(loss.item()) * weight_sum
        total_policy_loss += float(p_loss.item()) * weight_sum
        total_value_loss += float(v_loss.item()) * weight_sum
        total_weight += weight_sum

        predicted = logits.argmax(dim=1)
        top1 += int((predicted == batch["best"]).sum().item())
        for row_idx, gold in enumerate(batch["best"].tolist()):
            valid_count = int(batch["move_mask"][row_idx].sum().item())
            order = torch.argsort(logits[row_idx, :valid_count], descending=True).tolist()
            best_rank_sum += order.index(int(gold)) + 1

        value_abs_error += float((values - batch["value"]).abs().sum().item())
        value_sign_correct += int((torch.sign(values) == torch.sign(batch["value"])).sum().item())

    return {
        "loss": total_loss / max(1e-12, total_weight),
        "policy_loss": total_policy_loss / max(1e-12, total_weight),
        "value_loss": total_value_loss / max(1e-12, total_weight),
        "top1": top1 / len(records),
        "avg_best_rank": best_rank_sum / len(records),
        "value_mae": value_abs_error / len(records),
        "value_sign_acc": value_sign_correct / len(records),
    }


class PolicyValueModel:
    def __init__(self, model: PolicyValueNet, *, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self._cache: dict[tuple[Any, ...], tuple[list[float], float]] = {}

    @property
    def value_scale(self) -> float:
        return self.model.config.value_scale

    @property
    def target_komi(self) -> int:
        return self.model.config.target_komi

    def clear_cache(self) -> None:
        self._cache.clear()

    def score_record(self, record: dict[str, Any]) -> list[float]:
        logits, _value = self.predict_record(record)
        return logits

    def predict_record(self, record: dict[str, Any]) -> tuple[list[float], float]:
        return self.predict_records([record])[0]

    def predict_records(self, records: list[dict[str, Any]]) -> list[tuple[list[float], float]]:
        if not records:
            return []

        batch_size = len(records)
        move_counts = [len(record["moves"]) for record in records]
        max_moves = max(1, max(move_counts))
        boards = torch.zeros(
            batch_size,
            len(BOARD_CHANNEL_NAMES),
            BOARD_SIZE,
            BOARD_SIZE,
            dtype=torch.float32,
            device=self.device,
        )
        move_features = torch.zeros(
            batch_size,
            max_moves,
            len(FEATURE_NAMES),
            dtype=torch.float32,
            device=self.device,
        )
        move_mask = torch.zeros(batch_size, max_moves, dtype=torch.bool, device=self.device)

        for idx, record in enumerate(records):
            boards[idx] = torch.tensor(
                board_channels_for_state(record["state"]),
                dtype=torch.float32,
                device=self.device,
            )
            if move_counts[idx]:
                features = feature_matrix(record)
                move_features[idx, : move_counts[idx]] = torch.tensor(
                    features,
                    dtype=torch.float32,
                    device=self.device,
                )
                move_mask[idx, : move_counts[idx]] = True

        with torch.no_grad():
            logits, values = self.model(boards, move_features, move_mask)

        output: list[tuple[list[float], float]] = []
        logits_cpu = logits.detach().cpu()
        values_cpu = values.detach().cpu()
        for idx, move_count in enumerate(move_counts):
            output.append(
                (
                    [float(x) for x in logits_cpu[idx, :move_count].tolist()],
                    float(values_cpu[idx].item()),
                )
            )
        return output

    def _state_cache_key(self, state: State, moves: tuple[Move, ...]) -> tuple[Any, ...]:
        move_key = tuple((move.row, move.col, move.value) for move in moves)
        return (
            *state.key(),
            state.first_score,
            state.second_score,
            state.moves_played,
            move_key,
        )

    def predict_state(self, state: State, moves: tuple[Move, ...] | None = None) -> tuple[list[float], float]:
        moves = legal_moves(state) if moves is None else moves
        key = self._state_cache_key(state, moves)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        record = record_for_state_moves(state, moves)
        prediction = self.predict_record(record)
        self._cache[key] = prediction
        return prediction

    def predict_states_batch(
        self,
        items: list[tuple[State, tuple[Move, ...]]],
    ) -> list[tuple[list[float], float]]:
        if not items:
            return []

        results: list[tuple[list[float], float] | None] = [None for _item in items]
        missed_indices: list[int] = []
        missed_records: list[dict[str, Any]] = []
        missed_keys: list[tuple[Any, ...]] = []

        for idx, (state, moves) in enumerate(items):
            key = self._state_cache_key(state, moves)
            cached = self._cache.get(key)
            if cached is not None:
                results[idx] = cached
                continue
            missed_indices.append(idx)
            missed_keys.append(key)
            missed_records.append(record_for_state_moves(state, moves))

        predictions = self.predict_records(missed_records)
        if len(predictions) != len(missed_records):
            raise RuntimeError(
                "policy-value batch prediction count mismatch: "
                f"requested={len(missed_records)} predicted={len(predictions)}"
            )
        for idx, key, prediction in zip(missed_indices, missed_keys, predictions):
            self._cache[key] = prediction
            results[idx] = prediction

        missing = [idx for idx, item in enumerate(results) if item is None]
        if missing:
            raise RuntimeError(f"missing policy-value predictions for batch indices: {missing}")
        return [item for item in results if item is not None]

    def value_raw(self, state: State, moves: tuple[Move, ...] | None = None) -> float:
        _logits, value_norm = self.predict_state(state, moves)
        return denormalize_value(value_norm, self.value_scale)

    def policy_priors(self, state: State, moves: tuple[Move, ...] | None = None) -> list[float]:
        moves = legal_moves(state) if moves is None else moves
        logits, _value = self.predict_state(state, moves)
        if not logits:
            return []
        best = max(logits)
        exps = [math.exp(max(-80.0, min(80.0, score - best))) for score in logits]
        total = sum(exps)
        return [value / total for value in exps]

    def save(
        self,
        path: Path,
        *,
        metrics: dict[str, float] | None = None,
        epoch: int | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "policy_value_net",
            "version": 1,
            "board_channel_names": list(BOARD_CHANNEL_NAMES),
            "move_feature_names": list(FEATURE_NAMES),
            "config": asdict(self.model.config),
            "metrics": metrics or {},
            "epoch": epoch,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in self.model.state_dict().items()
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, *, device: torch.device | str = "cpu") -> "PolicyValueModel":
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except TypeError:  # pragma: no cover - older torch compatibility
            payload = torch.load(path, map_location=device)
        if payload.get("model_type") != "policy_value_net":
            raise ValueError(f"not a policy-value checkpoint: {path}")
        if tuple(payload["board_channel_names"]) != BOARD_CHANNEL_NAMES:
            raise ValueError("board channel names do not match this code version")
        if tuple(payload["move_feature_names"]) != FEATURE_NAMES:
            raise ValueError("move feature names do not match this code version")
        config = PolicyValueConfig(**payload["config"])
        model = PolicyValueNet(config)
        model.load_state_dict(payload["state_dict"])
        return cls(model, device=device)
