from __future__ import annotations

from typing import Any, Iterable

from .core import Move, Player, State, initial_state


def move_to_record(move: Move) -> dict[str, int]:
    return {
        "row": move.row,
        "col": move.col,
        "value": move.value,
        "index": move.index,
        "ones": move.ones,
    }


def move_from_record(record: dict[str, Any]) -> Move:
    return Move(
        row=int(record["row"]),
        col=int(record["col"]),
        value=int(record["value"]),
    )


def state_to_record(state: State) -> dict[str, Any]:
    return {
        "values": list(state.values),
        "owners": list(state.owners),
        "current": int(state.current),
        "first_score": state.first_score,
        "second_score": state.second_score,
        "moves_played": state.moves_played,
        "remaining": state.remaining(),
    }


def state_from_record(record: dict[str, Any]) -> State:
    return State(
        values=tuple(int(x) for x in record["values"]),
        owners=tuple(int(x) for x in record["owners"]),
        current=Player(int(record["current"])),
        first_score=int(record["first_score"]),
        second_score=int(record["second_score"]),
        moves_played=int(record["moves_played"]),
    )


def replay_game_states(game: dict[str, Any]) -> Iterable[tuple[State, dict[str, Any]]]:
    state = initial_state()
    for move_record in game["moves"]:
        yield state, move_record
        state = state.apply(move_from_record(move_record))
