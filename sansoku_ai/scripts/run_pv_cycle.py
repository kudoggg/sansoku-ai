from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter


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
            if line.strip():
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    continue
                count += 1
    return count


def copy_model(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"champion={target} source={source}", flush=True)


def read_arena(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def arena_line(payload: dict) -> str:
    by_side = payload.get("by_side") or {}
    first = by_side.get("first") or {}
    second = by_side.get("second") or {}
    first_line = (
        f"first={int(first.get('wins', 0))}-{int(first.get('losses', 0))}-"
        f"{int(first.get('draws', 0))} avg={float(first.get('avg_margin', 0.0)):+.2f}"
    )
    second_line = (
        f"second={int(second.get('wins', 0))}-{int(second.get('losses', 0))}-"
        f"{int(second.get('draws', 0))} avg={float(second.get('avg_margin', 0.0)):+.2f}"
    )
    return (
        f"games={int(payload['games'])} komi={int(payload.get('komi', 0))} "
        f"wins={int(payload['wins'])} losses={int(payload['losses'])} "
        f"draws={int(payload['draws'])} avg_margin={float(payload['avg_margin']):+.2f} "
        f"| {first_line}; {second_line}"
    )


def should_promote(
    payload: dict,
    *,
    min_margin: float,
    min_win_rate: float,
    min_second_margin: float | None,
) -> tuple[bool, list[str]]:
    games = int(payload["games"])
    wins = int(payload["wins"])
    losses = int(payload["losses"])
    draws = int(payload["draws"])
    avg_margin = float(payload["avg_margin"])
    win_rate = (wins + 0.5 * draws) / max(1, games)
    reasons = [
        f"wins>{losses}: {wins > losses}",
        f"avg_margin>{min_margin:+.2f}: {avg_margin > min_margin}",
        f"win_rate>={min_win_rate:.3f}: {win_rate >= min_win_rate}",
    ]
    ok = wins > losses and avg_margin > min_margin and win_rate >= min_win_rate

    if min_second_margin is not None:
        second = (payload.get("by_side") or {}).get("second") or {}
        second_avg = float(second.get("avg_margin", 0.0))
        second_ok = second_avg >= min_second_margin
        reasons.append(f"second_avg>={min_second_margin:+.2f}: {second_ok}")
        ok = ok and second_ok

    return ok, reasons


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def cycle_index_from_name(name: str) -> int | None:
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def require_even_games(value: int, name: str) -> None:
    if value > 0 and value % 2 != 0:
        raise SystemExit(f"{name} must be even so first/second games are balanced")


def print_step_summary(step_logs: list[tuple[str, str, float]]) -> None:
    print("step_summary:")
    for name, status, elapsed in step_logs:
        print(f"  {name}: {status} elapsed={elapsed:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run policy/value self-play cycles. Each cycle generates PUCT self-play "
            "with the current champion, trains a new policy/value net from MCTS visit "
            "policies, then promotes it only if it beats the previous champion."
        )
    )
    parser.add_argument("--prefix", default="pv_cycle")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--champion-model", type=Path, default=None)
    parser.add_argument("--reset-champion", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("data/pv_iterations"))
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--player", default="puct100")
    parser.add_argument("--puct-simulations", type=int, default=100)
    parser.add_argument("--puct-batch-size", type=int, default=8)
    parser.add_argument("--puct-leaf-depth", type=int, default=0)
    parser.add_argument("--puct-leaf-weight", type=float, default=0.0)
    parser.add_argument("--puct-leaf-move-limit", type=int, default=8)
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--endgame", type=int, default=4)
    parser.add_argument("--komi", type=int, default=16)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--opening-top-k", type=int, default=8)
    parser.add_argument("--opening-temperature", type=float, default=4.0)
    parser.add_argument("--visit-sampling-plies", type=int, default=12)
    parser.add_argument("--visit-temperature", type=float, default=1.0)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-noise-fraction", type=float, default=0.25)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--policy-target-mode", choices=("best", "policy"), default="policy")
    parser.add_argument("--value-target-mode", choices=("search", "final", "blend"), default="search")
    parser.add_argument("--select-metric", choices=("loss", "top1", "value"), default="loss")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-symmetry-augment", action="store_true")
    parser.add_argument("--include-non-mcts", action="store_true")
    parser.add_argument(
        "--extra-source",
        action="append",
        default=[],
        help=(
            "Additional alpha-beta reanalysis source for PV training, formatted as "
            "PATH:TIER:WEIGHT:QUALITY. Can be repeated."
        ),
    )
    parser.add_argument("--extra-policy-temperature", type=float, default=6.0)
    parser.add_argument("--extra-min-analyzed-moves", type=int, default=2)
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-fixed-arena", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--arena-candidate", default="puct80")
    parser.add_argument("--arena-games", type=int, default=20)
    parser.add_argument("--arena-strong-games", type=int, default=20)
    parser.add_argument("--arena-full-games", type=int, default=2)
    parser.add_argument("--arena-strong-full-games", type=int, default=2)
    parser.add_argument("--promote-candidate", default="puct100")
    parser.add_argument("--promote-games", type=int, default=40)
    parser.add_argument("--promote-min-margin", type=float, default=0.0)
    parser.add_argument("--promote-min-win-rate", type=float, default=0.5)
    parser.add_argument("--promote-min-second-margin", type=float, default=None)
    parser.add_argument("--continue-on-fail", action="store_true")
    parser.add_argument("--disable-promotion-mining", action="store_true")
    parser.add_argument("--mined-limit", type=int, default=300)
    parser.add_argument("--mined-high-value", type=int, default=10)
    parser.add_argument("--mined-depth", type=int, default=5)
    parser.add_argument("--mined-root-limit", type=int, default=16)
    parser.add_argument("--mined-move-limit", type=int, default=12)
    parser.add_argument("--mined-weight", type=float, default=7.0)
    parser.add_argument("--mined-quality", type=int, default=45)
    parser.add_argument("--reanalyze-workers", type=int, default=1)
    args = parser.parse_args()

    if args.cycles < 1:
        raise SystemExit("--cycles must be at least 1")
    require_even_games(args.arena_games, "--arena-games")
    require_even_games(args.arena_strong_games, "--arena-strong-games")
    require_even_games(args.arena_full_games, "--arena-full-games")
    require_even_games(args.arena_strong_full_games, "--arena-strong-full-games")
    require_even_games(args.promote_games, "--promote-games")

    py = sys.executable
    champion = args.champion_model or Path("models") / f"policy_value_{args.prefix}_champion.pt"
    summary_path = args.summary or args.output_root / f"{args.prefix}_cycle_summary.json"

    if args.reset_champion or not champion.exists():
        if not args.initial_model.exists():
            raise SystemExit(f"missing initial model: {args.initial_model}")
        copy_model(args.initial_model, champion)
    else:
        print(f"resume champion={champion}", flush=True)

    summary: dict = {
        "prefix": args.prefix,
        "komi": args.komi,
        "champion_model": str(champion),
        "initial_model": str(args.initial_model),
        "cycles": [],
    }
    if summary_path.exists() and not args.reset_champion:
        try:
            old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if old_summary.get("prefix") == args.prefix:
                summary["cycles"] = old_summary.get("cycles", [])
        except json.JSONDecodeError:
            pass

    carry_sources: list[str] = list(args.extra_source)
    carry_sources.extend(
        str(cycle["mined_source_for_next_cycle"])
        for cycle in summary.get("cycles", [])
        if cycle.get("mined_source_for_next_cycle")
        and (cycle_index_from_name(str(cycle.get("name", ""))) or 0) < args.start_index
    )
    puct_leaf_args = [
        "--puct-leaf-depth",
        str(args.puct_leaf_depth),
        "--puct-leaf-weight",
        str(args.puct_leaf_weight),
        "--puct-leaf-move-limit",
        str(args.puct_leaf_move_limit),
    ]

    for offset in range(args.cycles):
        cycle_index = args.start_index + offset
        name = f"{args.prefix}_{cycle_index:03d}"
        run_dir = args.output_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        cycle_seed = args.seed + cycle_index * 100000
        selfplay = run_dir / "selfplay.jsonl"
        dataset = run_dir / "training_dataset.jsonl"
        train = run_dir / "train.jsonl"
        val = run_dir / "val.jsonl"
        model = Path("models") / f"policy_value_{name}.pt"
        promotion_arena = run_dir / "arena_new_vs_champion.json"
        step_logs: list[tuple[str, str, float]] = []

        print(
            f"\ncycle_start name={name} champion={champion} seed={cycle_seed}",
            flush=True,
        )
        if carry_sources:
            print("extra_sources " + " ".join(carry_sources), flush=True)

        existing_games = count_jsonl(selfplay)
        generate_cmd = [
            py,
            "-m",
            "sansoku_ai.scripts.generate_pv_selfplay",
            "--model",
            str(champion),
            "--output",
            str(selfplay),
            "--games",
            str(args.games),
            "--player",
            args.player,
            "--seed",
            str(cycle_seed),
            "--device",
            args.device,
            "--endgame",
            str(args.endgame),
            "--komi",
            str(args.komi),
            "--puct-simulations",
            str(args.puct_simulations),
            "--puct-batch-size",
            str(args.puct_batch_size),
            *puct_leaf_args,
            "--cpuct",
            str(args.cpuct),
            "--root-dirichlet-alpha",
            str(args.root_dirichlet_alpha),
            "--root-noise-fraction",
            str(args.root_noise_fraction),
            "--opening-plies",
            str(args.opening_plies),
            "--opening-top-k",
            str(args.opening_top_k),
            "--opening-temperature",
            str(args.opening_temperature),
            "--visit-sampling-plies",
            str(args.visit_sampling_plies),
            "--visit-temperature",
            str(args.visit_temperature),
            "--progress-every",
            str(max(1, args.games // 10)),
        ]
        if existing_games >= args.games and not args.force:
            print(f"skip existing {selfplay} games={existing_games}/{args.games}", flush=True)
            log_step(step_logs, "generate pv selfplay", "skipped", 0.0)
        else:
            if selfplay.exists() and not args.force:
                generate_cmd.append("--resume")
            run_step(step_logs, "generate pv selfplay", generate_cmd)

        build_cmd = [
            py,
            "-m",
            "sansoku_ai.scripts.build_pv_training_dataset",
            str(selfplay),
            "--output",
            str(dataset),
            "--train-output",
            str(train),
            "--val-output",
            str(val),
            "--seed",
            str(cycle_seed),
            "--tier",
            name,
            "--quality",
            "40",
            "--sample-weight",
            "1.0",
            "--komi",
            str(args.komi),
            "--target-komi",
            "0",
        ]
        if not args.no_symmetry_augment:
            build_cmd.append("--symmetry-augment")
        if args.include_non_mcts:
            build_cmd.append("--include-non-mcts")
        for source in carry_sources:
            build_cmd.extend(["--extra-source", source])
        if carry_sources:
            build_cmd.extend(
                [
                    "--extra-policy-temperature",
                    str(args.extra_policy_temperature),
                    "--extra-min-analyzed-moves",
                    str(args.extra_min_analyzed_moves),
                ]
            )
        if args.keep_duplicates:
            build_cmd.append("--keep-duplicates")
        if dataset.exists() and train.exists() and val.exists() and not args.force:
            print(f"skip existing {dataset} train={train} val={val}", flush=True)
            log_step(step_logs, "build pv dataset", "skipped", 0.0)
        else:
            run_step(step_logs, "build pv dataset", build_cmd)

        train_cmd = [
            py,
            "-m",
            "sansoku_ai.scripts.train_policy_value",
            "--train",
            str(train),
            "--val",
            str(val),
            "--init-model",
            str(champion),
            "--epochs",
            str(args.train_epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--value-weight",
            str(args.value_weight),
            "--policy-target-mode",
            args.policy_target_mode,
            "--value-target-mode",
            args.value_target_mode,
            "--select-metric",
            args.select_metric,
            "--device",
            args.device,
            "--output",
            str(model),
        ]
        maybe_run(step_logs, "train policy value", train_cmd, model, force=args.force)

        run_step(
            step_logs,
            "evaluate policy value",
            [
                py,
                "-m",
                "sansoku_ai.scripts.evaluate_policy_value",
                str(model),
                str(val),
                "--batch-size",
                str(args.batch_size),
                "--policy-target-mode",
                args.policy_target_mode,
                "--value-target-mode",
                args.value_target_mode,
                "--value-weight",
                str(args.value_weight),
                "--device",
                args.device,
            ],
        )

        if not args.skip_fixed_arena and args.arena_games > 0:
            maybe_run(
                step_logs,
                "arena puct vs ab23 limited",
                [
                    py,
                    "-m",
                    "sansoku_ai.scripts.arena",
                    "--candidate",
                    args.arena_candidate,
                    "--opponent-mix",
                    "ab2:0.5,ab3:0.5",
                    "--games",
                    str(args.arena_games),
                    "--seed",
                    str(cycle_seed + 1000),
                    "--policy-value-model",
                    str(model),
                    "--endgame",
                    str(args.endgame),
                    "--puct-simulations",
                    str(args.puct_simulations),
                    "--puct-batch-size",
                    str(args.puct_batch_size),
                    *puct_leaf_args,
                    "--komi",
                    str(args.komi),
                    "--output",
                    str(run_dir / "arena_puct_vs_ab23_limited.json"),
                    "--progress-every",
                    str(max(1, args.arena_games // 2)),
                ],
                run_dir / "arena_puct_vs_ab23_limited.json",
                force=args.force,
            )

        if not args.skip_fixed_arena and args.arena_strong_games > 0:
            maybe_run(
                step_logs,
                "arena puct vs ab34 limited",
                [
                    py,
                    "-m",
                    "sansoku_ai.scripts.arena",
                    "--candidate",
                    args.arena_candidate,
                    "--opponent-mix",
                    "ab3:0.5,ab4:0.5",
                    "--games",
                    str(args.arena_strong_games),
                    "--seed",
                    str(cycle_seed + 2000),
                    "--policy-value-model",
                    str(model),
                    "--endgame",
                    str(args.endgame),
                    "--puct-simulations",
                    str(args.puct_simulations),
                    "--puct-batch-size",
                    str(args.puct_batch_size),
                    *puct_leaf_args,
                    "--komi",
                    str(args.komi),
                    "--output",
                    str(run_dir / "arena_puct_vs_ab34_limited.json"),
                    "--progress-every",
                    str(max(1, args.arena_strong_games // 2)),
                ],
                run_dir / "arena_puct_vs_ab34_limited.json",
                force=args.force,
            )

        if not args.skip_fixed_arena and args.arena_full_games > 0:
            maybe_run(
                step_logs,
                "arena puct vs ab23 full",
                [
                    py,
                    "-m",
                    "sansoku_ai.scripts.arena",
                    "--candidate",
                    args.arena_candidate,
                    "--opponent-mix",
                    "ab2:0.5,ab3:0.5",
                    "--games",
                    str(args.arena_full_games),
                    "--seed",
                    str(cycle_seed + 3000),
                    "--policy-value-model",
                    str(model),
                    "--endgame",
                    str(args.endgame),
                    "--puct-simulations",
                    str(args.puct_simulations),
                    "--puct-batch-size",
                    str(args.puct_batch_size),
                    *puct_leaf_args,
                    "--full-candidate",
                    "--full-opponent",
                    "--komi",
                    str(args.komi),
                    "--output",
                    str(run_dir / "arena_puct_vs_ab23_full.json"),
                    "--progress-every",
                    str(max(1, args.arena_full_games // 2)),
                ],
                run_dir / "arena_puct_vs_ab23_full.json",
                force=args.force,
            )

        if not args.skip_fixed_arena and args.arena_strong_full_games > 0:
            maybe_run(
                step_logs,
                "arena puct vs ab34 full",
                [
                    py,
                    "-m",
                    "sansoku_ai.scripts.arena",
                    "--candidate",
                    args.arena_candidate,
                    "--opponent-mix",
                    "ab3:0.5,ab4:0.5",
                    "--games",
                    str(args.arena_strong_full_games),
                    "--seed",
                    str(cycle_seed + 4000),
                    "--policy-value-model",
                    str(model),
                    "--endgame",
                    str(args.endgame),
                    "--puct-simulations",
                    str(args.puct_simulations),
                    "--puct-batch-size",
                    str(args.puct_batch_size),
                    *puct_leaf_args,
                    "--full-candidate",
                    "--full-opponent",
                    "--komi",
                    str(args.komi),
                    "--output",
                    str(run_dir / "arena_puct_vs_ab34_full.json"),
                    "--progress-every",
                    str(max(1, args.arena_strong_full_games // 2)),
                ],
                run_dir / "arena_puct_vs_ab34_full.json",
                force=args.force,
            )

        promotion_cmd = [
            py,
            "-m",
            "sansoku_ai.scripts.arena",
            "--candidate",
            args.promote_candidate,
            "--opponent-mix",
            f"{args.promote_candidate}:1",
            "--games",
            str(args.promote_games),
            "--seed",
            str(cycle_seed + 50000),
            "--policy-value-model",
            str(model),
            "--opponent-policy-value-model",
            str(champion),
            "--endgame",
            str(args.endgame),
            "--puct-simulations",
            str(args.puct_simulations),
            "--puct-batch-size",
            str(args.puct_batch_size),
            *puct_leaf_args,
            "--komi",
            str(args.komi),
            "--output",
            str(promotion_arena),
            "--progress-every",
            str(max(1, args.promote_games // 2)),
            "--record-games",
            "--record-losses-only",
        ]
        maybe_run(step_logs, "arena new vs champion", promotion_cmd, promotion_arena, force=args.force)

        promotion_payload = read_arena(promotion_arena)
        promoted, reasons = should_promote(
            promotion_payload,
            min_margin=args.promote_min_margin,
            min_win_rate=args.promote_min_win_rate,
            min_second_margin=args.promote_min_second_margin,
        )
        print(f"promotion_arena {arena_line(promotion_payload)}", flush=True)
        print("promotion_reasons " + "; ".join(reasons), flush=True)
        print(f"promotion_decision promoted={promoted}", flush=True)

        mined_source = None
        mined_positions = run_dir / "mined_promotion_positions.jsonl"
        mined_d5 = run_dir / "mined_promotion_d5.jsonl"
        if not args.disable_promotion_mining and int(promotion_payload["losses"]) > 0:
            maybe_run(
                step_logs,
                "mine promotion losses",
                [
                    py,
                    "-m",
                    "sansoku_ai.scripts.mine_arena_hard_positions",
                    str(promotion_arena),
                    "--output",
                    str(mined_positions),
                    "--limit",
                    str(args.mined_limit),
                    "--high-value",
                    str(args.mined_high_value),
                    "--id-prefix",
                    name,
                    "--include-all-candidate-loss-moves",
                    "--include-opponent-turns",
                ],
                mined_positions,
                force=args.force,
            )
            mined_count = count_jsonl(mined_positions)
            print(f"promotion_mining positions={mined_count} path={mined_positions}", flush=True)
            if mined_count > 0:
                reanalyze_mined_cmd = [
                    py,
                    "-m",
                    "sansoku_ai.scripts.reanalyze_positions",
                    str(mined_positions),
                    "--output",
                    str(mined_d5),
                    "--depth",
                    str(args.mined_depth),
                    "--endgame",
                    str(args.endgame),
                    "--root-limit",
                    str(args.mined_root_limit),
                    "--move-limit",
                    str(args.mined_move_limit),
                    "--workers",
                    str(args.reanalyze_workers),
                    "--progress-every",
                    str(max(1, mined_count // 2)),
                ]
                if mined_d5.exists() and not args.force:
                    reanalyze_mined_cmd.append("--resume")
                maybe_run(
                    step_logs,
                    "reanalyze promotion mining",
                    reanalyze_mined_cmd,
                    mined_d5,
                    force=args.force,
                )
                mined_source = (
                    f"{mined_d5}:{name}_promotion_mined:"
                    f"{args.mined_weight:g}:{args.mined_quality}"
                )
                carry_sources.append(mined_source)
                print(f"next_cycle_extra_source {mined_source}", flush=True)

        if not args.skip_package:
            run([py, "-m", "sansoku_ai.scripts.package_iteration", name])

        print_step_summary(step_logs)
        cycle_record = {
            "name": name,
            "run_dir": str(run_dir),
            "previous_champion": str(champion),
            "new_model": str(model),
            "promotion_arena": str(promotion_arena),
            "promoted": promoted,
            "promotion": {
                "games": int(promotion_payload["games"]),
                "wins": int(promotion_payload["wins"]),
                "losses": int(promotion_payload["losses"]),
                "draws": int(promotion_payload["draws"]),
                "avg_margin": float(promotion_payload["avg_margin"]),
                "by_side": promotion_payload.get("by_side"),
                "reasons": reasons,
            },
            "mined_positions": str(mined_positions) if mined_positions.exists() else None,
            "mined_reanalysis": str(mined_d5) if mined_d5.exists() else None,
            "mined_source_for_next_cycle": mined_source,
            "step_logs": [
                {"name": step, "status": status, "elapsed_sec": elapsed}
                for step, status, elapsed in step_logs
            ],
        }
        summary["cycles"].append(cycle_record)

        if promoted:
            copy_model(model, champion)
            summary["champion_source"] = str(model)
            write_summary(summary_path, summary)
            continue

        write_summary(summary_path, summary)
        if not args.continue_on_fail:
            print(f"cycle_stop name={name} reason=promotion_failed", flush=True)
            break

    print("\ncycle complete")
    print(f"champion={champion}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
