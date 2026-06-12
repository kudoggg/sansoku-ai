from __future__ import annotations

import json
from pathlib import Path

from .ranker import LinearRanker, RankerModel


def load_ranker_model(path: Path) -> RankerModel:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_type = payload.get("model_type")
        if model_type == "linear_ranker":
            return LinearRanker.load(path)
        raise ValueError(f"unknown JSON ranker model_type={model_type!r}: {path}")

    if suffix in {".pt", ".pth"}:
        from .nn_ranker import NnRanker

        return NnRanker.load(path)

    try:
        return LinearRanker.load(path)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        from .nn_ranker import NnRanker

        return NnRanker.load(path)
