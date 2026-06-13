from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from sansoku_ai.core import Move, Player, State, legal_moves
from sansoku_ai.jsonl import iter_jsonl_records
from sansoku_ai.records import move_from_record, move_to_record, replay_game_states, state_to_record
from sansoku_ai.scripts.build_training_dataset import (
    SourceSpec,
    build_example as build_ab_example,
    canonical_symmetry_key,
    example_key,
    expand_records,
    move_key,
    parse_source,
    side_weight_summary,
    write_jsonl,
)
from sansoku_ai.scripts.sample_positions import phase_for


@dataclass(frozen=True)
class BuildStats:
    games: int
    positions_seen: int
    records: int
    skipped_opening_or_exact: int
    skipped_no_policy: int
    onehot_records: int


@dataclass(frozen=True)
class ExtraSourceStats:
    tier: str
    read: int
    kept: int
    skipped: int


def final_margin_for_current(game: dict[str, Any], state: State, *, komi: int) -> float:
    first_score = int(game.get("first_score", 0))
    second_score = int(game.get("second_score", 0))
    first_margin = first_score - second_score + komi
    return float(first_margin if state.current == Player.FIRST else -first_margin)


def stat_key(stat: dict[str, Any]) -> str:
    return f"{int(stat['row'])},{int(stat['col'])},{int(stat['value'])}"


def normalize_probs(values: list[float]) -> list[float]:
    total = sum(max(0.0, value) for value in values)
    if total <= 0.0:
        return [0.0 for _value in values]
    return [max(0.0, value) / total for value in values]


def move_to_training_item(
    move: Move,
    *,
    policy: float,
    stat: dict[str, Any] | None,
    value_scale: float,
) -> dict[str, Any]:
    item = {
        "move": move_to_record(move),
        "action_key": move_key(move_to_record(move)),
        "policy": policy,
    }
    if stat is not None:
        item["visits"] = int(stat.get("visits", 0))
        item["prior"] = float(stat.get("prior", 0.0))
        item["q"] = float(stat.get("q", 0.0))
        item["search_value"] = float(stat.get("q", 0.0)) * value_scale
    return item


def build_record_from_state(
    *,
    game: dict[str, Any],
    state: State,
    move_record: dict[str, Any],
    tier: str,
    quality: int,
    sample_weight: float,
    first_weight: float,
    second_weight: float,
    komi: int,
    target_komi: int,
    value_scale: float,
    include_non_mcts: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    moves = legal_moves(state)
    if not moves:
        return None, "no_legal_moves"

    stats = list(move_record.get("mcts_policy") or [])
    played_move = move_from_record(move_record)
    played_index = next((idx for idx, move in enumerate(moves) if move == played_move), None)
    if played_index is None:
        return None, "played_not_legal"

    policy_source = "mcts_visit"
    stats_by_key = {stat_key(stat): stat for stat in stats}
    raw_policy = [
        float((stats_by_key.get(move_key(move_to_record(move))) or {}).get("visit_policy", 0.0))
        for move in moves
    ]
    policy = normalize_probs(raw_policy)

    if sum(policy) <= 0.0:
        if not include_non_mcts:
            if str(move_record.get("policy", "")).startswith(("opening", "exact")):
                return None, "opening_or_exact"
            return None, "no_mcts_policy"
        policy = [0.0 for _move in moves]
        policy[played_index] = 1.0
        policy_source = "played_onehot"

    best_index = max(
        range(len(moves)),
        key=lambda idx: (policy[idx], moves[idx].value, -moves[idx].index),
    )
    best_move = moves[best_index]
    current_target = final_margin_for_current(game, state, komi=target_komi)
    side_multiplier = first_weight if state.current == Player.FIRST else second_weight
    weight = sample_weight * side_multiplier

    training_moves = [
        move_to_training_item(
            move,
            policy=prob,
            stat=stats_by_key.get(move_key(move_to_record(move))),
            value_scale=value_scale,
        )
        for move, prob in zip(moves, policy)
    ]

    record_id = f"g{game['game']}_p{state.moves_played}"
    return {
        "id": record_id,
        "tier": tier,
        "quality": quality,
        "sample_weight": weight,
        "state": state_to_record(state),
        "phase": phase_for(state.moves_played),
        "source_game": int(game["game"]),
        "source_ply": state.moves_played,
        "played_policy": str(move_record.get("policy", "")),
        "policy_source": policy_source,
        "played_move": move_to_record(played_move),
        "best_move": move_to_record(best_move),
        "best_index": best_index,
        "played_index": played_index,
        "best_value": current_target,
        "search_margin_target": current_target,
        "final_margin_target": current_target,
        "final_margin_first_minus_second": int(game.get("margin", 0)),
        "komi": komi,
        "target_komi": target_komi,
        "legal_count": len(moves),
        "analyzed_count": len(training_moves),
        "depth": None,
        "endgame": None,
        "root_limit": None,
        "move_limit": None,
        "exact": False,
        "policy_temperature": None,
        "moves": training_moves,
    }, None


def build_records_from_games(
    games: list[dict[str, Any]],
    *,
    tier: str,
    quality: int,
    sample_weight: float,
    first_weight: float,
    second_weight: float,
    komi: int,
    target_komi: int,
    value_scale: float,
    include_non_mcts: bool,
) -> tuple[list[dict[str, Any]], BuildStats]:
    records: list[dict[str, Any]] = []
    skipped_opening_or_exact = 0
    skipped_no_policy = 0
    onehot_records = 0
    positions_seen = 0

    for game in games:
        for state, move_record in replay_game_states(game):
            positions_seen += 1
            record, reason = build_record_from_state(
                game=game,
                state=state,
                move_record=move_record,
                tier=tier,
                quality=quality,
                sample_weight=sample_weight,
                first_weight=first_weight,
                second_weight=second_weight,
                komi=komi,
                target_komi=target_komi,
                value_scale=value_scale,
                include_non_mcts=include_non_mcts,
            )
            if record is None:
                if reason == "opening_or_exact":
                    skipped_opening_or_exact += 1
                else:
                    skipped_no_policy += 1
                continue
            if record["policy_source"] == "played_onehot":
                onehot_records += 1
            records.append(record)

    return records, BuildStats(
        games=len(games),
        positions_seen=positions_seen,
        records=len(records),
        skipped_opening_or_exact=skipped_opening_or_exact,
        skipped_no_policy=skipped_no_policy,
        onehot_records=onehot_records,
    )


def dedupe_records(records: list[dict[str, Any]], *, symmetry_augment: bool) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        key = canonical_symmetry_key(record) if symmetry_augment else example_key(record)
        old = by_key.get(key)
        if old is None or float(record.get("sample_weight", 1.0)) >= float(
            old.get("sample_weight", 1.0)
        ):
            by_key[key] = record
    return list(by_key.values())


def records_from_extra_sources(
    sources: list[SourceSpec],
    *,
    policy_temperature: float,
    min_analyzed_moves: int,
) -> tuple[list[dict[str, Any]], list[ExtraSourceStats]]:
    records: list[dict[str, Any]] = []
    stats: list[ExtraSourceStats] = []
    for source in sources:
        read = 0
        kept = 0
        skipped = 0
        for raw in iter_jsonl_records(source.path):
            read += 1
            record = build_ab_example(
                raw,
                source,
                policy_temperature=policy_temperature,
                min_analyzed_moves=min_analyzed_moves,
            )
            if record is None:
                skipped += 1
                continue
            records.append(record)
            kept += 1
        stats.append(ExtraSourceStats(tier=source.tier, read=read, kept=kept, skipped=skipped))
    return records, stats


def split_and_expand(
    records: list[dict[str, Any]],
    *,
    seed: int,
    val_ratio: float,
    symmetry_augment: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    base = list(records)
    base.sort(key=lambda item: str(item["id"]))
    rng.shuffle(base)
    val_count = int(round(len(base) * val_ratio))
    val_base = base[:val_count]
    train_base = base[val_count:]
    train = expand_records(train_base, symmetry_augment=symmetry_augment)
    val = expand_records(val_base, symmetry_augment=symmetry_augment)
    return train + val, train, val


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert policy/value self-play game logs into the same legal-move-list "
            "training format used by train_policy_value."
        )
    )
    parser.add_argument("games", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, default=None)
    parser.add_argument("--val-output", type=Path, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tier", default="puct_selfplay")
    parser.add_argument("--quality", type=int, default=20)
    parser.add_argument("--sample-weight", type=float, default=1.0)
    parser.add_argument("--first-weight", type=float, default=1.0)
    parser.add_argument("--second-weight", type=float, default=1.0)
    parser.add_argument("--komi", type=int, default=16)
    parser.add_argument(
        "--target-komi",
        type=int,
        default=0,
        help=(
            "Komi included in value targets. Keep 0 for current pvab/puct code, "
            "which adds komi correction during search."
        ),
    )
    parser.add_argument("--value-scale", type=float, default=80.0)
    parser.add_argument("--include-non-mcts", action="store_true")
    parser.add_argument(
        "--extra-source",
        type=parse_source,
        action="append",
        default=[],
        help=(
            "Additional alpha-beta reanalysis source as PATH:TIER:WEIGHT:QUALITY. "
            "Can be repeated. This is useful for mined promotion losses."
        ),
    )
    parser.add_argument("--extra-policy-temperature", type=float, default=6.0)
    parser.add_argument("--extra-min-analyzed-moves", type=int, default=2)
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--symmetry-augment", action="store_true")
    args = parser.parse_args()

    games = list(iter_jsonl_records(args.games))
    records, stats = build_records_from_games(
        games,
        tier=args.tier,
        quality=args.quality,
        sample_weight=args.sample_weight,
        first_weight=args.first_weight,
        second_weight=args.second_weight,
        komi=args.komi,
        target_komi=args.target_komi,
        value_scale=args.value_scale,
        include_non_mcts=args.include_non_mcts,
    )
    mcts_base_count = len(records)
    extra_records, extra_stats = records_from_extra_sources(
        args.extra_source,
        policy_temperature=args.extra_policy_temperature,
        min_analyzed_moves=args.extra_min_analyzed_moves,
    )
    records.extend(extra_records)
    base_count = len(records)
    if not args.keep_duplicates:
        records = dedupe_records(records, symmetry_augment=args.symmetry_augment)

    if args.train_output is not None or args.val_output is not None:
        all_records, train, val = split_and_expand(
            records,
            seed=args.seed,
            val_ratio=args.val_ratio,
            symmetry_augment=args.symmetry_augment,
        )
        write_jsonl(args.train_output or Path("data/pv_train.jsonl"), train)
        write_jsonl(args.val_output or Path("data/pv_val.jsonl"), val)
    else:
        all_records = expand_records(records, symmetry_augment=args.symmetry_augment)
        train = []
        val = []

    write_jsonl(args.output, all_records)

    weights = [float(item.get("sample_weight", 1.0)) for item in all_records]
    analyzed_counts = [int(item["analyzed_count"]) for item in all_records]
    print(
        f"games={stats.games} positions_seen={stats.positions_seen} "
        f"mcts_records={stats.records} onehot_records={stats.onehot_records} "
        f"skipped_opening_or_exact={stats.skipped_opening_or_exact} "
        f"skipped_no_policy={stats.skipped_no_policy}"
    )
    if extra_stats:
        extra_summary = ",".join(
            f"{item.tier}:read={item.read},kept={item.kept},skipped={item.skipped}"
            for item in extra_stats
        )
        print(f"extra_sources={len(extra_stats)} {extra_summary}")
    print(
        f"wrote={args.output} examples={len(all_records)} base_records={base_count} "
        f"mcts_base_records={mcts_base_count} extra_records={len(extra_records)} "
        f"deduped_records={len(records)} symmetry_augment={args.symmetry_augment}"
    )
    if train or val:
        print(f"train={len(train)} val={len(val)}")
    if all_records:
        print(
            f"sample_weight avg={mean(weights):.2f} min={min(weights):.2f} "
            f"max={max(weights):.2f}"
        )
        print(
            f"analyzed_count avg={mean(analyzed_counts):.2f} "
            f"min={min(analyzed_counts)} max={max(analyzed_counts)}"
        )
        print(side_weight_summary(all_records))


if __name__ == "__main__":
    main()
