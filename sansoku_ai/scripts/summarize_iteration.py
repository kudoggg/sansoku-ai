from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_score(stats: dict[str, Any]) -> str:
    return (
        f"{int(stats.get('wins', 0))}-"
        f"{int(stats.get('losses', 0))}-"
        f"{int(stats.get('draws', 0))}"
    )


def fmt_avg(stats: dict[str, Any]) -> str:
    return f"{float(stats.get('avg_margin', 0.0)):+.2f}"


def print_arena(path: Path) -> None:
    payload = load_json(path)
    name = path.stem
    overall = {
        "wins": payload.get("wins", 0),
        "losses": payload.get("losses", 0),
        "draws": payload.get("draws", 0),
        "avg_margin": payload.get("avg_margin", 0.0),
    }
    print(
        f"{name}: games={int(payload.get('games', 0))} "
        f"komi={int(payload.get('komi', 0))} score={fmt_score(overall)} avg={fmt_avg(overall)}"
    )
    by_side = payload.get("by_side", {})
    if by_side:
        first = by_side.get("first", {})
        second = by_side.get("second", {})
        print(
            f"  first score={fmt_score(first)} avg={fmt_avg(first)} | "
            f"second score={fmt_score(second)} avg={fmt_avg(second)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("iteration", type=Path, help="Iteration directory or name")
    args = parser.parse_args()

    run_dir = args.iteration
    if not run_dir.exists():
        run_dir = Path("data") / "iterations" / str(args.iteration)
    if not run_dir.exists():
        raise SystemExit(f"iteration not found: {args.iteration}")

    print(f"iteration={run_dir}")
    for path in sorted(run_dir.glob("arena_*.json")):
        print_arena(path)


if __name__ == "__main__":
    main()
