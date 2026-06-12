from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without torch
    raise ModuleNotFoundError(
        "PyTorch is required for NN rankers. Install torch before using "
        "train_nn_ranker/evaluate_nn_ranker or a .pt ranker model."
    ) from exc

from .core import BOARD_SIZE, EMPTY_OWNER, INITIAL_OWNER, Player
from .ranker import FEATURE_NAMES, best_index, clamp, feature_matrix, target_policy


BOARD_CHANNEL_NAMES = (
    "empty",
    "initial",
    "first",
    "second",
    "current_player_cells",
    "opponent_cells",
    "ones_norm",
    "value_norm",
    "current_is_second",
    "moves_played_norm",
    "remaining_norm",
    "first_score_norm",
    "second_score_norm",
    "current_margin_norm",
)


@dataclass(frozen=True)
class NnRankerConfig:
    board_channels: int = len(BOARD_CHANNEL_NAMES)
    move_features: int = len(FEATURE_NAMES)
    conv_channels: int = 32
    board_hidden: int = 64
    hidden: int = 128
    dropout: float = 0.0


def _score_margin_current(state: dict[str, Any]) -> float:
    first_score = float(state["first_score"])
    second_score = float(state["second_score"])
    current = int(state["current"])
    return first_score - second_score if current == int(Player.FIRST) else second_score - first_score


def board_channels_for_state(state: dict[str, Any]) -> list[list[list[float]]]:
    planes = [
        [[0.0 for _col in range(BOARD_SIZE)] for _row in range(BOARD_SIZE)]
        for _name in BOARD_CHANNEL_NAMES
    ]
    values = state["values"]
    owners = state["owners"]
    current = int(state["current"])
    opponent = int(Player.SECOND if current == int(Player.FIRST) else Player.FIRST)
    current_is_second = 1.0 if current == int(Player.SECOND) else 0.0
    moves_played_norm = clamp(float(state["moves_played"]) / 32.0, 0.0, 1.0)
    remaining_norm = clamp(float(state["remaining"]) / 32.0, 0.0, 1.0)
    first_score_norm = clamp(float(state["first_score"]) / 400.0, 0.0, 2.0)
    second_score_norm = clamp(float(state["second_score"]) / 400.0, 0.0, 2.0)
    current_margin_norm = clamp(_score_margin_current(state) / 200.0, -2.0, 2.0)

    for idx, owner_raw in enumerate(owners):
        row, col = divmod(idx, BOARD_SIZE)
        owner = int(owner_raw)
        value = int(values[idx])
        if owner == EMPTY_OWNER:
            planes[0][row][col] = 1.0
        elif owner == INITIAL_OWNER:
            planes[1][row][col] = 1.0
        elif owner == int(Player.FIRST):
            planes[2][row][col] = 1.0
        elif owner == int(Player.SECOND):
            planes[3][row][col] = 1.0

        if owner == current:
            planes[4][row][col] = 1.0
        elif owner == opponent:
            planes[5][row][col] = 1.0

        if owner != EMPTY_OWNER:
            planes[6][row][col] = (value % 10) / 9.0
            planes[7][row][col] = clamp(value / 81.0, 0.0, 2.0)

        planes[8][row][col] = current_is_second
        planes[9][row][col] = moves_played_norm
        planes[10][row][col] = remaining_norm
        planes[11][row][col] = first_score_norm
        planes[12][row][col] = second_score_norm
        planes[13][row][col] = current_margin_norm

    return planes


class BoardMoveRanker(nn.Module):
    def __init__(self, config: NnRankerConfig) -> None:
        super().__init__()
        self.config = config
        self.board_net = nn.Sequential(
            nn.Conv2d(config.board_channels, config.conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.conv_channels, config.conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(config.conv_channels * BOARD_SIZE * BOARD_SIZE, config.board_hidden),
            nn.ReLU(),
        )
        self.move_net = nn.Sequential(
            nn.Linear(config.board_hidden + config.move_features, config.hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, 1),
        )

    def forward(
        self,
        boards: torch.Tensor,
        move_features: torch.Tensor,
        move_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        board_embedding = self.board_net(boards)
        batch_size, max_moves, _feat = move_features.shape
        expanded = board_embedding[:, None, :].expand(batch_size, max_moves, -1)
        logits = self.move_net(torch.cat([expanded, move_features], dim=-1)).squeeze(-1)
        if move_mask is not None:
            logits = logits.masked_fill(~move_mask, -1e9)
        return logits


def collate_ranker_records(
    records: list[dict[str, Any]],
    *,
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
        weights[batch_idx] = min(float(record.get("sample_weight", 1.0)), max_sample_weight)

    return {
        "boards": boards,
        "move_features": moves,
        "move_mask": mask,
        "policy": policy,
        "best": best,
        "weights": weights,
    }


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def ranker_loss(
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
        raise ValueError(f"unknown target mode: {target_mode}")
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


@torch.no_grad()
def evaluate_nn_ranker_model(
    model: BoardMoveRanker,
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
    target_mode: str = "best",
    max_sample_weight: float = 5.0,
) -> dict[str, float]:
    if not records:
        return {"loss": 0.0, "top1": 0.0, "avg_best_rank": 0.0}

    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    top1 = 0
    best_rank_sum = 0.0

    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        batch = move_batch_to_device(
            collate_ranker_records(chunk, max_sample_weight=max_sample_weight),
            device,
        )
        logits = model(batch["boards"], batch["move_features"], batch["move_mask"])
        loss = ranker_loss(logits, batch, target_mode=target_mode)
        weight_sum = float(batch["weights"].sum().item())
        total_loss += float(loss.item()) * weight_sum
        total_weight += weight_sum

        predicted = logits.argmax(dim=1)
        top1 += int((predicted == batch["best"]).sum().item())
        for row_idx, gold in enumerate(batch["best"].tolist()):
            valid_count = int(batch["move_mask"][row_idx].sum().item())
            order = torch.argsort(logits[row_idx, :valid_count], descending=True).tolist()
            best_rank_sum += order.index(int(gold)) + 1

    return {
        "loss": total_loss / max(1e-12, total_weight),
        "top1": top1 / len(records),
        "avg_best_rank": best_rank_sum / len(records),
    }


class NnRanker:
    def __init__(self, model: BoardMoveRanker, *, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    def score_record(self, record: dict[str, Any]) -> list[float]:
        move_count = len(record["moves"])
        boards = torch.tensor(
            [board_channels_for_state(record["state"])],
            dtype=torch.float32,
            device=self.device,
        )
        move_features = torch.tensor(
            [feature_matrix(record)],
            dtype=torch.float32,
            device=self.device,
        )
        move_mask = torch.ones(1, move_count, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits = self.model(boards, move_features, move_mask)
        return [float(x) for x in logits[0, :move_count].detach().cpu().tolist()]

    def save(
        self,
        path: Path,
        *,
        metrics: dict[str, float] | None = None,
        epoch: int | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "nn_ranker",
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
    def load(cls, path: Path, *, device: torch.device | str = "cpu") -> "NnRanker":
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except TypeError:  # pragma: no cover - older torch compatibility
            payload = torch.load(path, map_location=device)
        if payload.get("model_type") != "nn_ranker":
            raise ValueError(f"not an nn_ranker checkpoint: {path}")
        if tuple(payload["board_channel_names"]) != BOARD_CHANNEL_NAMES:
            raise ValueError("board channel names do not match this code version")
        if tuple(payload["move_feature_names"]) != FEATURE_NAMES:
            raise ValueError("move feature names do not match this code version")
        config = NnRankerConfig(**payload["config"])
        model = BoardMoveRanker(config)
        model.load_state_dict(payload["state_dict"])
        return cls(model, device=device)
