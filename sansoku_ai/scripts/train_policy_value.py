from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sansoku_ai.policy_value import (
    PolicyValueConfig,
    PolicyValueModel,
    PolicyValueNet,
    collate_policy_value_records,
    evaluate_policy_value_model,
    move_batch_to_device,
    policy_value_loss,
)
from sansoku_ai.ranker import load_jsonl


def choose_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def train_epoch(
    model: PolicyValueNet,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    policy_target_mode: str,
    value_weight: float,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_weight = 0.0

    for batch_raw in loader:
        batch = move_batch_to_device(batch_raw, device)
        optimizer.zero_grad(set_to_none=True)
        logits, values = model(batch["boards"], batch["move_features"], batch["move_mask"])
        loss, p_loss, v_loss = policy_value_loss(
            logits,
            values,
            batch,
            policy_target_mode=policy_target_mode,
            value_weight=value_weight,
        )
        loss.backward()
        optimizer.step()
        weight_sum = float(batch["weights"].sum().item())
        total_loss += float(loss.item()) * weight_sum
        total_policy_loss += float(p_loss.item()) * weight_sum
        total_value_loss += float(v_loss.item()) * weight_sum
        total_weight += weight_sum

    denom = max(1e-12, total_weight)
    return total_loss / denom, total_policy_loss / denom, total_value_loss / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train_v2.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("data/val_v2.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("models/policy_value_v1.pt"))
    parser.add_argument(
        "--init-model",
        type=Path,
        default=None,
        help="Continue training from an existing policy-value checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--conv-channels", type=int, default=48)
    parser.add_argument("--board-hidden", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--value-scale", type=float, default=80.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--max-sample-weight", type=float, default=5.0)
    parser.add_argument("--policy-target-mode", choices=("best", "policy"), default="policy")
    parser.add_argument("--value-target-mode", choices=("search", "final", "blend"), default="search")
    parser.add_argument(
        "--select-metric",
        choices=("loss", "top1", "value"),
        default="loss",
        help="Which validation metric selects the saved checkpoint.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = choose_device(args.device)
    train_records = load_jsonl(args.train)
    val_records = load_jsonl(args.val)
    if args.init_model is not None:
        pv = PolicyValueModel.load(args.init_model, device=device)
        model = pv.model
        config = model.config
    else:
        config = PolicyValueConfig(
            conv_channels=args.conv_channels,
            board_hidden=args.board_hidden,
            hidden=args.hidden,
            dropout=args.dropout,
            value_scale=args.value_scale,
        )
        model = PolicyValueNet(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    rng = random.Random(args.seed)

    def collate(records: list[dict]):
        return collate_policy_value_records(
            records,
            value_target_mode=args.value_target_mode,
            value_scale=config.value_scale,
            max_sample_weight=args.max_sample_weight,
        )

    loader = DataLoader(
        train_records,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    print(
        f"train={len(train_records)} val={len(val_records)} output={args.output} "
        f"device={device} config={config} policy_target={args.policy_target_mode} "
        f"value_target={args.value_target_mode} value_weight={args.value_weight} "
        f"init_model={args.init_model}"
    )
    initial = evaluate_policy_value_model(
        model,
        val_records,
        batch_size=args.batch_size,
        device=device,
        policy_target_mode=args.policy_target_mode,
        value_target_mode=args.value_target_mode,
        value_weight=args.value_weight,
        max_sample_weight=args.max_sample_weight,
    )
    print(
        f"initial_val loss={initial['loss']:.4f} policy_loss={initial['policy_loss']:.4f} "
        f"value_loss={initial['value_loss']:.4f} top1={initial['top1']:.3f} "
        f"rank={initial['avg_best_rank']:.2f} value_mae={initial['value_mae']:.3f} "
        f"value_sign={initial['value_sign_acc']:.3f}"
    )

    best_score = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    best_metrics = initial

    def score(metrics: dict[str, float]) -> float:
        if args.select_metric == "loss":
            return metrics["loss"]
        if args.select_metric == "top1":
            return -metrics["top1"]
        return metrics["value_loss"]

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_records)
        train_loss, train_policy_loss, train_value_loss = train_epoch(
            model,
            loader,
            optimizer=optimizer,
            device=device,
            policy_target_mode=args.policy_target_mode,
            value_weight=args.value_weight,
        )
        train_metrics = evaluate_policy_value_model(
            model,
            train_records[: min(len(train_records), 2000)],
            batch_size=args.batch_size,
            device=device,
            policy_target_mode=args.policy_target_mode,
            value_target_mode=args.value_target_mode,
            value_weight=args.value_weight,
            max_sample_weight=args.max_sample_weight,
        )
        val_metrics = evaluate_policy_value_model(
            model,
            val_records,
            batch_size=args.batch_size,
            device=device,
            policy_target_mode=args.policy_target_mode,
            value_target_mode=args.value_target_mode,
            value_weight=args.value_weight,
            max_sample_weight=args.max_sample_weight,
        )
        val_score = score(val_metrics)
        if val_score < best_score:
            best_score = val_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            best_metrics = val_metrics

        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"train_policy={train_policy_loss:.4f} train_value={train_value_loss:.4f} "
            f"train_top1={train_metrics['top1']:.3f} train_rank={train_metrics['avg_best_rank']:.2f} "
            f"val_loss={val_metrics['loss']:.4f} val_policy={val_metrics['policy_loss']:.4f} "
            f"val_value={val_metrics['value_loss']:.4f} val_top1={val_metrics['top1']:.3f} "
            f"val_rank={val_metrics['avg_best_rank']:.2f} "
            f"val_value_mae={val_metrics['value_mae']:.3f} "
            f"val_value_sign={val_metrics['value_sign_acc']:.3f}"
        )

    model.load_state_dict(best_state)
    PolicyValueModel(model, device=device).save(args.output, metrics=best_metrics, epoch=best_epoch)
    print(
        f"saved={args.output} best_epoch={best_epoch} best_val_loss={best_metrics['loss']:.4f} "
        f"best_val_policy={best_metrics['policy_loss']:.4f} "
        f"best_val_value={best_metrics['value_loss']:.4f} "
        f"best_val_top1={best_metrics['top1']:.3f} "
        f"best_val_rank={best_metrics['avg_best_rank']:.2f} "
        f"best_val_value_mae={best_metrics['value_mae']:.3f} "
        f"best_val_value_sign={best_metrics['value_sign_acc']:.3f}"
    )


if __name__ == "__main__":
    main()
