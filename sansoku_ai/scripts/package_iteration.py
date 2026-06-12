from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def add_if_exists(archive: tarfile.TarFile, path: Path) -> int:
    if not path.exists():
        print(f"missing {path}")
        return 0
    archive.add(path, arcname=str(path))
    print(f"added {path}")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Iteration name, e.g. iter_runpod_002_quick")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--include-data", action="store_true")
    args = parser.parse_args()

    output = args.output or Path("outputs") / f"{args.name}_artifacts.tar.gz"
    output.parent.mkdir(parents=True, exist_ok=True)

    models = (
        Path("models") / f"linear_ranker_{args.name}.json",
        Path("models") / f"nn_ranker_{args.name}.pt",
    )
    log = Path("logs") / f"{args.name}.log"
    run_dir = Path("data") / "iterations" / args.name
    arena_files = sorted(run_dir.glob("arena_*.json")) if run_dir.exists() else []
    mined_files = sorted(run_dir.glob("mined_*.jsonl")) if run_dir.exists() else []

    added = 0
    with tarfile.open(output, "w:gz") as archive:
        for model in models:
            added += add_if_exists(archive, model)
        added += add_if_exists(archive, log)
        for path in arena_files:
            added += add_if_exists(archive, path)
        for path in mined_files:
            added += add_if_exists(archive, path)
        if args.include_data:
            added += add_if_exists(archive, run_dir)

    print(f"wrote {output} files={added}")


if __name__ == "__main__":
    main()
