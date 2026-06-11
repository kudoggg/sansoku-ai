from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_SOURCES = (
    "data/reanalyzed_all_d3_fast.jsonl:d3_fast:1:10",
    "data/hard_500_d4.jsonl:d4_hard:2:20",
    "data/hard_100_d5_root16_move12.jsonl:d5_root16_move12:4:30",
    "data/hard_50_d6_root24_move12.jsonl:d6_root24_move12:10:50",
    "data/reanalyzed_ru_all_d3_fast.jsonl:ru_d3_fast:1:12",
    "data/hard_ru_500_d5_root16_move12.jsonl:ru_d5_root16_move12:5:35",
)


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def maybe_run(cmd: list[str], output: Path, *, force: bool) -> None:
    if output.exists() and not force:
        print(f"skip existing {output}")
        return
    run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Iteration name, e.g. iter003")
    parser.add_argument("--output-root", type=Path, default=Path("data/iterations"))
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy-mix",
        default="ab2:0.3,ab3:0.4,ru2:0.15,ru3:0.15",
    )
    parser.add_argument("--ranker-model", type=Path, default=Path("models/linear_ranker_v2.json"))
    parser.add_argument("--endgame", type=int, default=4)
    parser.add_argument("--move-limit", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--hard-limit", type=int, default=500)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--no-base-sources", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-arena", action="store_true")
    parser.add_argument("--arena-games", type=int, default=20)
    parser.add_argument("--arena-strong-games", type=int, default=0)
    parser.add_argument("--arena-full-games", type=int, default=0)
    parser.add_argument("--arena-strong-full-games", type=int, default=0)
    args = parser.parse_args()

    run_dir = args.output_root / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    mixed = run_dir / "mixed.jsonl"
    positions = run_dir / "positions.jsonl"
    reanalyzed_d3 = run_dir / "reanalyzed_d3_fast.jsonl"
    hard = run_dir / "hard_from_d3.jsonl"
    hard_d5 = run_dir / "hard_d5_root16_move12.jsonl"
    dataset = run_dir / "training_dataset.jsonl"
    train = run_dir / "train.jsonl"
    val = run_dir / "val.jsonl"
    model = Path("models") / f"linear_ranker_{args.name}.json"

    py = sys.executable

    maybe_run(
        [
            py,
            "-m",
            "sansoku_ai.scripts.generate_mixed_games",
            "--games",
            str(args.games),
            "--seed",
            str(args.seed),
            "--endgame",
            str(args.endgame),
            "--move-limit",
            str(args.move_limit),
            "--policy-mix",
            args.policy_mix,
            "--ranker-model",
            str(args.ranker_model),
            "--output",
            str(mixed),
            "--progress-every",
            str(max(1, args.games // 10)),
        ],
        mixed,
        force=args.force,
    )

    maybe_run(
        [
            py,
            "-m",
            "sansoku_ai.scripts.sample_positions",
            str(mixed),
            "--output",
            str(positions),
            "--id-prefix",
            args.name,
        ],
        positions,
        force=args.force,
    )

    reanalyze_d3_cmd = [
        py,
        "-m",
        "sansoku_ai.scripts.reanalyze_positions",
        str(positions),
        "--output",
        str(reanalyzed_d3),
        "--depth",
        "3",
        "--endgame",
        "4",
        "--root-limit",
        "8",
        "--move-limit",
        "8",
        "--workers",
        str(args.workers),
        "--progress-every",
        "500",
    ]
    if reanalyzed_d3.exists() and not args.force:
        reanalyze_d3_cmd.append("--resume")
    run(reanalyze_d3_cmd)

    maybe_run(
        [
            py,
            "-m",
            "sansoku_ai.scripts.select_hard_positions",
            str(reanalyzed_d3),
            "--output",
            str(hard),
            "--limit",
            str(args.hard_limit),
        ],
        hard,
        force=args.force,
    )

    reanalyze_d5_cmd = [
        py,
        "-m",
        "sansoku_ai.scripts.reanalyze_positions",
        str(hard),
        "--output",
        str(hard_d5),
        "--depth",
        "5",
        "--endgame",
        "4",
        "--root-limit",
        "16",
        "--move-limit",
        "12",
        "--workers",
        str(args.workers),
        "--progress-every",
        "50",
    ]
    if hard_d5.exists() and not args.force:
        reanalyze_d5_cmd.append("--resume")
    run(reanalyze_d5_cmd)

    build_cmd = [
        py,
        "-m",
        "sansoku_ai.scripts.build_training_dataset",
        "--output",
        str(dataset),
        "--train-output",
        str(train),
        "--val-output",
        str(val),
    ]
    if not args.no_base_sources:
        for source in BASE_SOURCES:
            if Path(source.split(":", 1)[0]).exists():
                build_cmd.extend(["--source", source])
    build_cmd.extend(
        [
            "--source",
            f"{reanalyzed_d3}:{args.name}_d3_fast:1:12",
            "--source",
            f"{hard_d5}:{args.name}_d5_root16_move12:5:35",
        ]
    )
    maybe_run(build_cmd, dataset, force=args.force)

    if not args.skip_train:
        maybe_run(
            [
                py,
                "-m",
                "sansoku_ai.scripts.train_linear_ranker",
                "--train",
                str(train),
                "--val",
                str(val),
                "--epochs",
                str(args.train_epochs),
                "--target-mode",
                "best",
                "--select-metric",
                "top1",
                "--lr",
                str(args.lr),
                "--output",
                str(model),
            ],
            model,
            force=args.force,
        )
        run(
            [
                py,
                "-m",
                "sansoku_ai.scripts.evaluate_ranker",
                str(model),
                str(val),
            ]
        )

    arena_model = model if model.exists() else args.ranker_model
    if not args.skip_arena and args.arena_games > 0:
        maybe_run(
            [
                py,
                "-m",
                "sansoku_ai.scripts.arena",
                "--candidate",
                "ru3",
                "--opponent-mix",
                "ab2:0.5,ab3:0.5",
                "--games",
                str(args.arena_games),
                "--seed",
                str(args.seed + 1000),
                "--ranker-model",
                str(arena_model),
                "--endgame",
                str(args.endgame),
                "--candidate-move-limit",
                str(args.move_limit),
                "--opponent-move-limit",
                str(args.move_limit),
                "--output",
                str(run_dir / "arena_ru3_vs_ab23_limited.json"),
                "--progress-every",
                str(max(1, args.arena_games // 2)),
            ],
            run_dir / "arena_ru3_vs_ab23_limited.json",
            force=args.force,
        )

    if not args.skip_arena and args.arena_strong_games > 0:
        maybe_run(
            [
                py,
                "-m",
                "sansoku_ai.scripts.arena",
                "--candidate",
                "ru3",
                "--opponent-mix",
                "ab3:0.5,ab4:0.5",
                "--games",
                str(args.arena_strong_games),
                "--seed",
                str(args.seed + 2000),
                "--ranker-model",
                str(arena_model),
                "--endgame",
                str(args.endgame),
                "--candidate-move-limit",
                str(args.move_limit),
                "--opponent-move-limit",
                str(args.move_limit),
                "--output",
                str(run_dir / "arena_ru3_vs_ab34_limited.json"),
                "--progress-every",
                str(max(1, args.arena_strong_games // 2)),
            ],
            run_dir / "arena_ru3_vs_ab34_limited.json",
            force=args.force,
        )

    if not args.skip_arena and args.arena_full_games > 0:
        maybe_run(
            [
                py,
                "-m",
                "sansoku_ai.scripts.arena",
                "--candidate",
                "ru3",
                "--opponent-mix",
                "ab2:0.5,ab3:0.5",
                "--games",
                str(args.arena_full_games),
                "--seed",
                str(args.seed + 3000),
                "--ranker-model",
                str(arena_model),
                "--endgame",
                str(args.endgame),
                "--full-candidate",
                "--full-opponent",
                "--output",
                str(run_dir / "arena_ru3_vs_ab23_full.json"),
                "--progress-every",
                str(max(1, args.arena_full_games // 2)),
            ],
            run_dir / "arena_ru3_vs_ab23_full.json",
            force=args.force,
        )

    if not args.skip_arena and args.arena_strong_full_games > 0:
        maybe_run(
            [
                py,
                "-m",
                "sansoku_ai.scripts.arena",
                "--candidate",
                "ru3",
                "--opponent-mix",
                "ab3:0.5,ab4:0.5",
                "--games",
                str(args.arena_strong_full_games),
                "--seed",
                str(args.seed + 4000),
                "--ranker-model",
                str(arena_model),
                "--endgame",
                str(args.endgame),
                "--full-candidate",
                "--full-opponent",
                "--output",
                str(run_dir / "arena_ru3_vs_ab34_full.json"),
                "--progress-every",
                str(max(1, args.arena_strong_full_games // 2)),
            ],
            run_dir / "arena_ru3_vs_ab34_full.json",
            force=args.force,
        )

    print(f"\niteration complete: {args.name}")
    print(f"run_dir={run_dir}")
    print(f"model={model}")


if __name__ == "__main__":
    main()
