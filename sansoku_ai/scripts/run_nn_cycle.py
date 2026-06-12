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


def run_if_missing(cmd: list[str], output: Path, *, force: bool) -> float:
    if output.exists() and not force:
        print(f"skip existing {output}", flush=True)
        return 0.0
    return run(cmd)


def read_arena(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_model(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"champion={target} source={source}", flush=True)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated NN-ranker bootstrap cycles. Each cycle generates games "
            "with the current champion, reanalyzes positions with alpha-beta, trains "
            "a new NN ranker, then promotes it only if it beats the previous champion."
        )
    )
    parser.add_argument("--prefix", default="nn_cycle")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--initial-ranker-model", type=Path, default=Path("models/nn_ranker_v2.pt"))
    parser.add_argument("--champion-model", type=Path, default=None)
    parser.add_argument("--reset-champion", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("data/iterations"))
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy-mix",
        default="ab2:0.15,ab3:0.30,ru2:0.25,ru3:0.30",
    )
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
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-fixed-arena", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--arena-games", type=int, default=40)
    parser.add_argument("--arena-strong-games", type=int, default=40)
    parser.add_argument("--arena-full-games", type=int, default=4)
    parser.add_argument("--arena-strong-full-games", type=int, default=2)
    parser.add_argument("--komi", type=int, default=16)
    parser.add_argument("--promote-games", type=int, default=60)
    parser.add_argument("--promote-min-margin", type=float, default=0.0)
    parser.add_argument("--promote-min-win-rate", type=float, default=0.5)
    parser.add_argument("--promote-min-second-margin", type=float, default=None)
    parser.add_argument("--promote-full", action="store_true")
    parser.add_argument("--continue-on-fail", action="store_true")
    args = parser.parse_args()

    if args.cycles < 1:
        raise SystemExit("--cycles must be at least 1")
    if args.promote_games < 2:
        raise SystemExit("--promote-games should be at least 2")

    py = sys.executable
    champion = args.champion_model or Path("models") / f"nn_ranker_{args.prefix}_champion.pt"
    summary_path = args.summary or args.output_root / f"{args.prefix}_cycle_summary.json"

    if args.reset_champion or not champion.exists():
        if not args.initial_ranker_model.exists():
            raise SystemExit(f"missing initial ranker model: {args.initial_ranker_model}")
        copy_model(args.initial_ranker_model, champion)
    else:
        print(f"resume champion={champion}", flush=True)

    summary: dict = {
        "prefix": args.prefix,
        "komi": args.komi,
        "champion_model": str(champion),
        "initial_ranker_model": str(args.initial_ranker_model),
        "cycles": [],
    }
    if summary_path.exists() and not args.reset_champion:
        try:
            old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if old_summary.get("prefix") == args.prefix:
                summary["cycles"] = old_summary.get("cycles", [])
        except json.JSONDecodeError:
            pass

    for offset in range(args.cycles):
        cycle_index = args.start_index + offset
        name = f"{args.prefix}_{cycle_index:03d}"
        run_dir = args.output_root / name
        model = Path("models") / f"nn_ranker_{name}.pt"
        promotion_arena = run_dir / "arena_new_vs_champion.json"
        cycle_seed = args.seed + cycle_index * 100000

        print(
            f"\ncycle_start name={name} champion={champion} seed={cycle_seed}",
            flush=True,
        )

        iteration_cmd = [
            py,
            "-m",
            "sansoku_ai.scripts.run_nn_iteration",
            "--name",
            name,
            "--output-root",
            str(args.output_root),
            "--games",
            str(args.games),
            "--seed",
            str(cycle_seed),
            "--policy-mix",
            args.policy_mix,
            "--ranker-model",
            str(champion),
            "--endgame",
            str(args.endgame),
            "--move-limit",
            str(args.move_limit),
            "--workers",
            str(args.workers),
            "--hard-limit",
            str(args.hard_limit),
            "--train-epochs",
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
            "--arena-games",
            "0" if args.skip_fixed_arena else str(args.arena_games),
            "--arena-strong-games",
            "0" if args.skip_fixed_arena else str(args.arena_strong_games),
            "--arena-full-games",
            "0" if args.skip_fixed_arena else str(args.arena_full_games),
            "--arena-strong-full-games",
            "0" if args.skip_fixed_arena else str(args.arena_strong_full_games),
            "--komi",
            str(args.komi),
            "--skip-package",
        ]
        if args.no_base_sources:
            iteration_cmd.append("--no-base-sources")
        if args.keep_duplicates:
            iteration_cmd.append("--keep-duplicates")
        if args.force:
            iteration_cmd.append("--force")

        run(iteration_cmd)
        if not model.exists():
            raise SystemExit(f"iteration did not produce model: {model}")

        promotion_cmd = [
            py,
            "-m",
            "sansoku_ai.scripts.arena",
            "--candidate",
            "ru3",
            "--opponent-mix",
            "ru3:1",
            "--games",
            str(args.promote_games),
            "--seed",
            str(cycle_seed + 50000),
            "--ranker-model",
            str(model),
            "--opponent-ranker-model",
            str(champion),
            "--endgame",
            str(args.endgame),
            "--candidate-move-limit",
            str(args.move_limit),
            "--opponent-move-limit",
            str(args.move_limit),
            "--komi",
            str(args.komi),
            "--output",
            str(promotion_arena),
            "--progress-every",
            str(max(1, args.promote_games // 2)),
        ]
        if args.promote_full:
            promotion_cmd.extend(["--full-candidate", "--full-opponent"])

        run_if_missing(promotion_cmd, promotion_arena, force=args.force)
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

        if not args.skip_package:
            run([py, "-m", "sansoku_ai.scripts.package_iteration", name])

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
