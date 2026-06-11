from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterator


def iter_jsonl_records(
    path: Path,
    *,
    skip_bad: bool = True,
    warn: bool = True,
) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                if not skip_bad:
                    raise
                if warn:
                    print(
                        f"warning: skipped bad JSONL line {path}:{line_no}: {exc}",
                        file=sys.stderr,
                    )


def load_jsonl_records(
    path: Path,
    *,
    skip_bad: bool = True,
    warn: bool = True,
) -> list[dict[str, Any]]:
    return list(iter_jsonl_records(path, skip_bad=skip_bad, warn=warn))
