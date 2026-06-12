from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sansoku_ai.nn_ranker import (
    BoardMoveRanker,
    NnRanker,
    NnRankerConfig,
    collate_ranker_records,
    evaluate_nn_ranker_model,
    move_batch_to_device,
    ranker_loss,
)
from sansoku_ai.ranker import load_jsonl


def choose_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def train_epoch(
    model: BoardMoveRanker,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_mode: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_weight = 0.0

    for batch_raw in loader:
        batch = move_batch_to_device(batch_raw, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["boards"], batch["move_features"], batch["move_mask"])
        loss = ranker_loss(logits, batch, target_mode=target_mode)
        loss.backward()
        optimizer.step()
        weight_sum = float(batch["weights"].sum().item())
        total_loss += float(loss.item()) * weight_sum
        total_weight += weight_sum

    return total_loss / max(1e-12, total_weight)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train_v2.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("data/val_v2.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("models/nn_ranker.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--conv-channels", type=int, default=32)
    parser.add_argument("--board-hidden", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-sample-weight", type=float, default=5.0)
    parser.add_argument("--target-mode", choices=("best", "policy"), default="best")
    parser.add_argument("--select-metric", choices=("loss", "top1"), default="top1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = choose_device(args.device)
    train_records = load_jsonl(args.train)
    val_records = load_jsonl(args.val)
    config = NnRankerConfig(
        conv_channels=args.conv_channels,
        board_hidden=args.board_hidden,
        hidden=args.hidden,
        dropout=args.dropout,
    )
    model = BoardMoveRanker(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    rng = random.Random(args.seed)

    def collate(records: list[dict]):
        return collate_ranker_records(records, max_sample_weight=args.max_sample_weight)

    loader = DataLoader(
        train_records,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    print(
        f"train={len(train_records)} val={len(val_records)} output={args.output} "
        f"device={device} config={config}"
    )
    initial = evaluate_nn_ranker_model(
        model,
        val_records,
        batch_size=args.batch_size,
        device=device,
        target_mode=args.target_mode,
        max_sample_weight=args.max_sample_weight,
    )
    print(
        f"initial_val loss={initial['loss']:.4f} top1={initial['top1']:.3f} "
        f"avg_best_rank={initial['avg_best_rank']:.2f}"
    )

    best_loss = float("inf")
    best_top1 = -1.0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    best_metrics = initial

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_records)
        train_loss = train_epoch(
            model,
            loader,
            optimizer=optimizer,
            device=device,
            target_mode=args.target_mode,
        )
        train_metrics = evaluate_nn_ranker_model(
            model,
            train_records[: min(len(train_records), 2000)],
            batch_size=args.batch_size,
            device=device,
            target_mode=args.target_mode,
            max_sample_weight=args.max_sample_weight,
        )
        val_metrics = evaluate_nn_ranker_model(
            model,
            val_records,
            batch_size=args.batch_size,
            device=device,
            target_mode=args.target_mode,
            max_sample_weight=args.max_sample_weight,
        )
        improved = (
            val_metrics["loss"] < best_loss
            if args.select_metric == "loss"
            else val_metrics["top1"] > best_top1
        )
        if improved:
            best_loss = val_metrics["loss"]
            best_top1 = val_metrics["top1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            best_metrics = val_metrics

        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"train_top1={train_metrics['top1']:.3f} train_rank={train_metrics['avg_best_rank']:.2f} "
            f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1']:.3f} "
            f"val_rank={val_metrics['avg_best_rank']:.2f}"
        )

    model.load_state_dict(best_state)
    NnRanker(model, device=device).save(args.output, metrics=best_metrics, epoch=best_epoch)
    print(
        f"saved={args.output} best_epoch={best_epoch} "
        f"best_val_loss={best_metrics['loss']:.4f} "
        f"best_val_top1={best_metrics['top1']:.3f} "
        f"best_val_rank={best_metrics['avg_best_rank']:.2f}"
    )


if __name__ == "__main__":
    main()
