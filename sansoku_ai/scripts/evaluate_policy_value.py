from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sansoku_ai.policy_value import PolicyValueModel, evaluate_policy_value_model
from sansoku_ai.ranker import load_jsonl


def choose_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--policy-target-mode", choices=("best", "policy"), default="policy")
    parser.add_argument("--value-target-mode", choices=("search", "final", "blend"), default="search")
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--max-sample-weight", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    pv = PolicyValueModel.load(args.model, device=device)
    records = load_jsonl(args.data)
    metrics = evaluate_policy_value_model(
        pv.model,
        records,
        batch_size=args.batch_size,
        device=device,
        policy_target_mode=args.policy_target_mode,
        value_target_mode=args.value_target_mode,
        value_weight=args.value_weight,
        max_sample_weight=args.max_sample_weight,
    )
    print(
        f"records={len(records)} loss={metrics['loss']:.4f} "
        f"policy_loss={metrics['policy_loss']:.4f} value_loss={metrics['value_loss']:.4f} "
        f"top1={metrics['top1']:.3f} avg_best_rank={metrics['avg_best_rank']:.2f} "
        f"value_mae={metrics['value_mae']:.3f} value_sign={metrics['value_sign_acc']:.3f} "
        f"value_scale={pv.value_scale:.1f} device={device}"
    )


if __name__ == "__main__":
    main()
