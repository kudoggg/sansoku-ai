from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

from sansoku_ai.core import legal_moves
from sansoku_ai.records import move_to_record, state_from_record
from sansoku_ai.search import AlphaBetaSearch


def state_analysis_key(
    position: dict[str, Any],
    *,
    depth: int,
    endgame: int,
    root_limit: int,
    move_limit: int | None,
) -> str:
    state = position["state"]
    payload = {
        "values": state["values"],
        "owners": state["owners"],
        "current": state["current"],
        "first_score": state["first_score"],
        "second_score": state["second_score"],
        "moves_played": state["moves_played"],
        "depth": depth,
        "endgame": endgame,
        "root_limit": root_limit,
        "move_limit": move_limit,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(str(record.get("id", "")))
    done.discard("")
    return done


def analyze_position_task(
    position: dict[str, Any],
    *,
    depth: int,
    endgame: int,
    root_limit: int,
    move_limit: int | None,
) -> dict[str, Any]:
    state = state_from_record(position["state"])
    search = AlphaBetaSearch(
        depth=depth,
        endgame_exact_remaining=endgame,
        move_limit=move_limit,
    )
    analysis = search.analyze_root(state, root_limit=root_limit)
    legal_count = len(legal_moves(state))
    analyzed_moves = [
        {"move": move_to_record(item.move), "value": item.value}
        for item in analysis.moves
    ]

    return {
        "id": position["id"],
        "source_game": position["source_game"],
        "source_ply": position["source_ply"],
        "phase": position["phase"],
        "state": position["state"],
        "played_move": position["played_move"],
        "played_policy": position["played_policy"],
        "final_margin": position["final_margin"],
        "depth": depth,
        "endgame": endgame,
        "root_limit": root_limit,
        "move_limit": move_limit,
        "analysis_key": state_analysis_key(
            position,
            depth=depth,
            endgame=endgame,
            root_limit=root_limit,
            move_limit=move_limit,
        ),
        "legal_count": legal_count,
        "analyzed_count": len(analyzed_moves),
        "exact": analysis.exact,
        "nodes": analysis.nodes,
        "elapsed_sec": analysis.elapsed_sec,
        "best_move": move_to_record(analysis.best_move)
        if analysis.best_move is not None
        else None,
        "best_value": analysis.best_value,
        "moves": analyzed_moves,
    }


def load_tasks(
    path: Path,
    *,
    max_positions: int | None,
    done_ids: set[str],
    dedupe_state: bool,
    depth: int,
    endgame: int,
    root_limit: int,
    move_limit: int | None,
) -> tuple[list[dict[str, Any]], int]:
    tasks: list[dict[str, Any]] = []
    skipped = 0
    seen_keys: set[str] = set()
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            if max_positions is not None and len(tasks) >= max_positions:
                break
            if not line.strip():
                continue
            position = json.loads(line)
            if str(position["id"]) in done_ids:
                skipped += 1
                continue
            if dedupe_state:
                key = state_analysis_key(
                    position,
                    depth=depth,
                    endgame=endgame,
                    root_limit=root_limit,
                    move_limit=move_limit,
                )
                if key in seen_keys:
                    skipped += 1
                    continue
                seen_keys.add(key)
            tasks.append(position)
    return tasks, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("positions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--endgame", type=int, default=8)
    parser.add_argument("--root-limit", type=int, default=10)
    parser.add_argument("--move-limit", type=int, default=10)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dedupe-state", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    count = 0
    total_nodes = 0
    total_analyzed_moves = 0
    done_ids = load_done_ids(args.output) if args.resume else set()
    tasks, skipped = load_tasks(
        args.positions,
        max_positions=args.max_positions,
        done_ids=done_ids,
        dedupe_state=args.dedupe_state,
        depth=args.depth,
        endgame=args.endgame,
        root_limit=args.root_limit,
        move_limit=args.move_limit,
    )

    mode = "a" if args.resume else "w"
    print(
        f"tasks={len(tasks)} skipped={skipped} workers={args.workers} "
        f"resume={args.resume} output={args.output}"
    )

    with args.output.open(mode, encoding="utf-8") as dst:
        if args.workers <= 1:
            for position in tasks:
                record = analyze_position_task(
                    position,
                    depth=args.depth,
                    endgame=args.endgame,
                    root_limit=args.root_limit,
                    move_limit=args.move_limit,
                )
                total_nodes += int(record["nodes"])
                total_analyzed_moves += int(record["analyzed_count"])
                count += 1
                dst.write(json.dumps(record, separators=(",", ":")) + "\n")
                if args.progress_every and count % args.progress_every == 0:
                    dst.flush()
                    elapsed = perf_counter() - start
                    print(
                        f"positions={count} elapsed={elapsed:.2f}s "
                        f"pos_per_sec={count / elapsed:.3f} nodes={total_nodes}"
                    )
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        analyze_position_task,
                        position,
                        depth=args.depth,
                        endgame=args.endgame,
                        root_limit=args.root_limit,
                        move_limit=args.move_limit,
                    )
                    for position in tasks
                ]
                for future in as_completed(futures):
                    record = future.result()
                    total_nodes += int(record["nodes"])
                    total_analyzed_moves += int(record["analyzed_count"])
                    count += 1
                    dst.write(json.dumps(record, separators=(",", ":")) + "\n")
                    if args.progress_every and count % args.progress_every == 0:
                        dst.flush()
                        elapsed = perf_counter() - start
                        print(
                            f"positions={count} elapsed={elapsed:.2f}s "
                            f"pos_per_sec={count / elapsed:.3f} nodes={total_nodes}"
                        )

    elapsed = perf_counter() - start
    print(
        f"done positions={count} elapsed={elapsed:.2f}s "
        f"pos_per_sec={count / elapsed if elapsed else 0:.3f} "
        f"avg_nodes={total_nodes / count if count else 0:.1f} "
        f"avg_analyzed_moves={total_analyzed_moves / count if count else 0:.1f} "
        f"wrote={args.output}"
    )


if __name__ == "__main__":
    main()
