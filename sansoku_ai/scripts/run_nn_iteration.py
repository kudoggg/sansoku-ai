from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter


BASE_SOURCES = (
    "data/reanalyzed_all_d3_fast.jsonl:d3_fast:1:10",
    "data/hard_500_d4.jsonl:d4_hard:2:20",
    "data/hard_100_d5_root16_move12.jsonl:d5_root16_move12:4:30",
    "data/hard_50_d6_root24_move12.jsonl:d6_root24_move12:10:50",
    "data/reanalyzed_ru_all_d3_fast.jsonl:ru_d3_fast:1:12",
    "data/hard_ru_500_d5_root16_move12.jsonl:ru_d5_root16_move12:5:35",
)


def run(cmd: list[str]) -> float:
    print("\n$", " ".join(cmd), flush=True)
    start = perf_counter()
    subprocess.run(cmd, check=True)
    elapsed = perf_counter() - start
    print(f"$ done elapsed={elapsed:.2f}s", flush=True)
    return elapsed


def log_step(logs: list[tuple[str, str, float]], name: str, status: str, elapsed: float) -> None:
    logs.append((name, status, elapsed))


def run_step(logs: list[tuple[str, str, float]], name: str, cmd: list[str]) -> None:
    log_step(logs, name, "ran", run(cmd))


def maybe_run(
    logs: list[tuple[str, str, float]],
    name: str,
    cmd: list[str],
    output: Path,
    *,
    force: bool,
) -> None:
    if output.exists() and not force:
        print(f"skip existing {output}", flush=True)
        log_step(logs, name, "skipped", 0.0)
        return
    run_step(logs, name, cmd)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                continue
            count += 1
    return count


def print_arena_result(path: Path) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(
        f"arena {path.name}: games={payload['games']} "
        f"komi={int(payload.get('komi', 0))} "
        f"wins={payload['wins']} losses={payload['losses']} draws={payload['draws']} "
        f"avg_margin={float(payload['avg_margin']):+.2f}"
    )
    by_side = payload.get("by_side")
    if by_side:
        first = by_side.get("first", {})
        second = by_side.get("second", {})
        print(
            "  by_side "
            f"first={int(first.get('wins', 0))}-{int(first.get('losses', 0))}-"
            f"{int(first.get('draws', 0))} avg={float(first.get('avg_margin', 0.0)):+.2f}; "
            f"second={int(second.get('wins', 0))}-{int(second.get('losses', 0))}-"
            f"{int(second.get('draws', 0))} avg={float(second.get('avg_margin', 0.0)):+.2f}"
        )


def require_even_games(value: int, name: str) -> None:
    if value > 0 and value % 2 != 0:
        raise SystemExit(f"{name} must be even so first/second games are balanced")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Iteration name, e.g. nn_iter003")
    parser.add_argument("--output-root", type=Path, default=Path("data/iterations"))
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy-mix",
        default="ab2:0.15,ab3:0.30,ru2:0.25,ru3:0.30",
    )
    parser.add_argument("--ranker-model", type=Path, default=Path("models/nn_ranker_v2.pt"))
    parser.add_argument("--endgame", type=int, default=4)
    parser.add_argument("--move-limit", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--hard-limit", type=int, default=1000)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--target-mode", choices=("best", "policy"), default="best")
    parser.add_argument("--select-metric", choices=("loss", "top1"), default="top1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-base-sources", action="store_true")
    parser.add_argument("--no-symmetry-augment", action="store_true")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-arena", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--arena-games", type=int, default=40)
    parser.add_argument("--arena-strong-games", type=int, default=40)
    parser.add_argument("--arena-full-games", type=int, default=4)
    parser.add_argument("--arena-strong-full-games", type=int, default=2)
    parser.add_argument("--komi", type=int, default=16)
    args = parser.parse_args()

    require_even_games(args.arena_games, "--arena-games")
    require_even_games(args.arena_strong_games, "--arena-strong-games")
    require_even_games(args.arena_full_games, "--arena-full-games")
    require_even_games(args.arena_strong_full_games, "--arena-strong-full-games")

    step_logs: list[tuple[str, str, float]] = []
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
    model = Path("models") / f"nn_ranker_{args.name}.pt"

    py = sys.executable

    generate_cmd = [
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
        "--workers",
        str(args.workers),
    ]
    existing_games = count_jsonl(mixed)
    if mixed.exists() and not args.force and existing_games >= args.games:
        print(f"skip existing {mixed} games={existing_games}/{args.games}", flush=True)
        log_step(step_logs, "generate mixed games", "skipped", 0.0)
    else:
        if mixed.exists() and not args.force:
            generate_cmd.append("--resume")
        run_step(step_logs, "generate mixed games", generate_cmd)

    maybe_run(
        step_logs,
        "sample positions",
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
        "1000",
    ]
    if reanalyzed_d3.exists() and not args.force:
        reanalyze_d3_cmd.append("--resume")
    run_step(step_logs, "reanalyze d3 fast", reanalyze_d3_cmd)

    maybe_run(
        step_logs,
        "select hard positions",
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
        "100",
    ]
    if hard_d5.exists() and not args.force:
        reanalyze_d5_cmd.append("--resume")
    run_step(step_logs, "reanalyze d5 hard", reanalyze_d5_cmd)

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
        "--seed",
        str(args.seed),
    ]
    if args.keep_duplicates:
        build_cmd.append("--keep-duplicates")
    if not args.no_symmetry_augment:
        build_cmd.append("--symmetry-augment")
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
    maybe_run(step_logs, "build training dataset", build_cmd, dataset, force=args.force)

    if not args.skip_train:
        maybe_run(
            step_logs,
            "train nn ranker",
            [
                py,
                "-m",
                "sansoku_ai.scripts.train_nn_ranker",
                "--train",
                str(train),
                "--val",
                str(val),
                "--epochs",
                str(args.train_epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--target-mode",
                args.target_mode,
                "--select-metric",
                args.select_metric,
                "--device",
                args.device,
                "--output",
                str(model),
            ],
            model,
            force=args.force,
        )
        run_step(
            step_logs,
            "evaluate nn ranker",
            [
                py,
                "-m",
                "sansoku_ai.scripts.evaluate_nn_ranker",
                str(model),
                str(val),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
            ],
        )

    arena_model = model if model.exists() else args.ranker_model
    if not args.skip_arena and args.arena_games > 0:
        maybe_run(
            step_logs,
            "arena ru3 vs ab23 limited",
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
                "--komi",
                str(args.komi),
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
            step_logs,
            "arena ru3 vs ab34 limited",
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
                "--komi",
                str(args.komi),
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
            step_logs,
            "arena ru3 vs ab23 full",
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
                "--komi",
                str(args.komi),
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
            step_logs,
            "arena ru3 vs ab34 full",
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
                "--komi",
                str(args.komi),
                "--output",
                str(run_dir / "arena_ru3_vs_ab34_full.json"),
                "--progress-every",
                str(max(1, args.arena_strong_full_games // 2)),
            ],
            run_dir / "arena_ru3_vs_ab34_full.json",
            force=args.force,
        )

    if not args.skip_package:
        run_step(
            step_logs,
            "package iteration",
            [py, "-m", "sansoku_ai.scripts.package_iteration", args.name],
        )

    print(f"\niteration complete: {args.name}")
    print(f"run_dir={run_dir}")
    print(f"model={model}")
    print("step_summary:")
    for name, status, elapsed in step_logs:
        print(f"  {name}: {status} elapsed={elapsed:.2f}s")
    print_arena_result(run_dir / "arena_ru3_vs_ab23_limited.json")
    print_arena_result(run_dir / "arena_ru3_vs_ab34_limited.json")
    print_arena_result(run_dir / "arena_ru3_vs_ab23_full.json")
    print_arena_result(run_dir / "arena_ru3_vs_ab34_full.json")


if __name__ == "__main__":
    main()
