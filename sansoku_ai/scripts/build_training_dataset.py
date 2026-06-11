from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    tier: str
    weight: float
    quality: int


@dataclass(frozen=True)
class DatasetExample:
    key: str
    quality: int
    record: dict[str, Any]


DEFAULT_SOURCES = (
    SourceSpec(Path("data/reanalyzed_all_d3_fast.jsonl"), "d3_fast", 1.0, 10),
    SourceSpec(Path("data/hard_500_d4.jsonl"), "d4_hard", 2.0, 20),
    SourceSpec(Path("data/hard_100_d5_root16_move12.jsonl"), "d5_root16_move12", 4.0, 30),
    SourceSpec(Path("data/hard_50_d6_root16_move12.jsonl"), "d6_root16_move12", 8.0, 40),
    SourceSpec(Path("data/hard_50_d6_root24_move12.jsonl"), "d6_root24_move12", 10.0, 50),
)


def parse_source(spec: str) -> SourceSpec:
    try:
        path_text, tier, weight_text, quality_text = spec.rsplit(":", 3)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "source must be PATH:TIER:WEIGHT:QUALITY"
        ) from exc
    return SourceSpec(Path(path_text), tier, float(weight_text), int(quality_text))


def same_move(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if a is None or b is None:
        return False
    return (
        int(a["row"]) == int(b["row"])
        and int(a["col"]) == int(b["col"])
        and int(a["value"]) == int(b["value"])
    )


def move_key(move: dict[str, Any]) -> str:
    return f"{int(move['row'])},{int(move['col'])},{int(move['value'])}"


def softmax(values: list[float], temperature: float) -> list[float]:
    if not values:
        return []
    if temperature <= 0:
        best_index = max(range(len(values)), key=lambda idx: values[idx])
        return [1.0 if idx == best_index else 0.0 for idx in range(len(values))]
    best = max(values)
    exps = [math.exp((value - best) / temperature) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def current_player_margin(record: dict[str, Any]) -> int:
    final_margin = int(record.get("final_margin", 0))
    current = int(record["state"]["current"])
    return final_margin if current == 1 else -final_margin


def build_example(
    raw: dict[str, Any],
    source: SourceSpec,
    *,
    policy_temperature: float,
    min_analyzed_moves: int,
) -> dict[str, Any] | None:
    moves = raw.get("moves", [])
    if len(moves) < min_analyzed_moves:
        return None

    values = [float(item["value"]) for item in moves]
    probs = softmax(values, policy_temperature)
    best_move = raw.get("best_move")
    played_move = raw.get("played_move")
    best_index = None
    played_index = None

    training_moves: list[dict[str, Any]] = []
    for idx, (item, prob) in enumerate(zip(moves, probs)):
        move = item["move"]
        if same_move(move, best_move):
            best_index = idx
        if same_move(move, played_move):
            played_index = idx
        training_moves.append(
            {
                "move": move,
                "action_key": move_key(move),
                "search_value": float(item["value"]),
                "policy": prob,
            }
        )

    if best_index is None:
        return None

    best_value = float(raw["best_value"])
    final_margin_current = current_player_margin(raw)
    search_margin_target = best_value
    final_margin_target = float(final_margin_current)

    return {
        "id": raw["id"],
        "tier": source.tier,
        "quality": source.quality,
        "sample_weight": source.weight,
        "state": raw["state"],
        "phase": raw.get("phase"),
        "source_game": raw.get("source_game"),
        "source_ply": raw.get("source_ply"),
        "played_policy": raw.get("played_policy"),
        "played_move": played_move,
        "best_move": best_move,
        "best_index": best_index,
        "played_index": played_index,
        "best_value": best_value,
        "search_margin_target": search_margin_target,
        "final_margin_target": final_margin_target,
        "final_margin_first_minus_second": int(raw.get("final_margin", 0)),
        "legal_count": int(raw.get("legal_count", 0)),
        "analyzed_count": len(training_moves),
        "depth": raw.get("depth"),
        "endgame": raw.get("endgame"),
        "root_limit": raw.get("root_limit"),
        "move_limit": raw.get("move_limit"),
        "exact": bool(raw.get("exact", False)),
        "policy_temperature": policy_temperature,
        "moves": training_moves,
    }


def example_key(record: dict[str, Any]) -> str:
    state = record["state"]
    payload = {
        "values": state["values"],
        "owners": state["owners"],
        "current": state["current"],
        "first_score": state["first_score"],
        "second_score": state["second_score"],
        "moves_played": state["moves_played"],
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=parse_source,
        action="append",
        help="PATH:TIER:WEIGHT:QUALITY. Can be repeated. Defaults to known data files.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/training_dataset.jsonl"))
    parser.add_argument("--train-output", type=Path, default=None)
    parser.add_argument("--val-output", type=Path, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-temperature", type=float, default=6.0)
    parser.add_argument("--min-analyzed-moves", type=int, default=2)
    parser.add_argument("--keep-duplicates", action="store_true")
    args = parser.parse_args()

    sources = args.source or [source for source in DEFAULT_SOURCES if source.path.exists()]
    if not sources:
        raise SystemExit("no sources found; pass --source PATH:TIER:WEIGHT:QUALITY")

    examples_by_key: dict[str, DatasetExample] = {}
    examples_list: list[DatasetExample] = []
    read_counts: dict[str, int] = {}
    kept_counts: dict[str, int] = {}
    skipped = 0

    for source in sources:
        read_counts[source.tier] = 0
        kept_counts[source.tier] = 0
        with source.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                read_counts[source.tier] += 1
                raw = json.loads(line)
                record = build_example(
                    raw,
                    source,
                    policy_temperature=args.policy_temperature,
                    min_analyzed_moves=args.min_analyzed_moves,
                )
                if record is None:
                    skipped += 1
                    continue
                key = example_key(record)
                example = DatasetExample(key=key, quality=source.quality, record=record)
                if args.keep_duplicates:
                    examples_list.append(example)
                    kept_counts[source.tier] += 1
                    continue

                old = examples_by_key.get(key)
                if old is None or example.quality >= old.quality:
                    examples_by_key[key] = example

    if not args.keep_duplicates:
        examples_list = list(examples_by_key.values())
        for example in examples_list:
            kept_counts[str(example.record["tier"])] = kept_counts.get(str(example.record["tier"]), 0) + 1

    rng = random.Random(args.seed)
    examples = [example.record for example in examples_list]
    examples.sort(key=lambda item: str(item["id"]))
    rng.shuffle(examples)

    write_jsonl(args.output, examples)

    if args.train_output is not None or args.val_output is not None:
        val_count = int(round(len(examples) * args.val_ratio))
        val = examples[:val_count]
        train = examples[val_count:]
        write_jsonl(args.train_output or Path("data/train.jsonl"), train)
        write_jsonl(args.val_output or Path("data/val.jsonl"), val)

    weights = [float(item["sample_weight"]) for item in examples]
    analyzed_counts = [int(item["analyzed_count"]) for item in examples]
    tier_summary = ",".join(f"{tier}:{kept_counts[tier]}" for tier in sorted(kept_counts))
    read_summary = ",".join(f"{tier}:{read_counts[tier]}" for tier in sorted(read_counts))
    print(f"sources={len(sources)} read={read_summary}")
    print(
        f"wrote={args.output} examples={len(examples)} skipped={skipped} "
        f"tiers={tier_summary}"
    )
    if examples:
        print(
            f"sample_weight avg={mean(weights):.2f} "
            f"min={min(weights):.2f} max={max(weights):.2f}"
        )
        print(
            f"analyzed_count avg={mean(analyzed_counts):.2f} "
            f"min={min(analyzed_counts)} max={max(analyzed_counts)}"
        )


if __name__ == "__main__":
    main()
