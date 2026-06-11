from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import Iterable


BOARD_SIZE = 6
CELL_COUNT = BOARD_SIZE * BOARD_SIZE
PLAYABLE_CELLS = CELL_COUNT - 4


class Player(IntEnum):
    FIRST = 1
    SECOND = 2

    def other(self) -> "Player":
        return Player.SECOND if self == Player.FIRST else Player.FIRST


EMPTY_OWNER = 0
INITIAL_OWNER = 3


@dataclass(frozen=True, order=True)
class Move:
    row: int
    col: int
    value: int

    @property
    def index(self) -> int:
        return self.row * BOARD_SIZE + self.col

    @property
    def ones(self) -> int:
        return self.value % 10


@dataclass(frozen=True)
class State:
    values: tuple[int, ...]
    owners: tuple[int, ...]
    current: Player
    first_score: int = 0
    second_score: int = 0
    moves_played: int = 0

    def remaining(self) -> int:
        return PLAYABLE_CELLS - self.moves_played

    def is_terminal(self) -> bool:
        return self.remaining() <= 0 or not legal_moves(self)

    def margin_for(self, player: Player, komi: int = 0) -> int:
        first_margin = self.first_score - self.second_score + komi
        return first_margin if player == Player.FIRST else -first_margin

    def apply(self, move: Move) -> "State":
        idx = move.index
        if self.owners[idx] != EMPTY_OWNER:
            raise ValueError(f"cell is not empty: ({move.row}, {move.col})")

        values = list(self.values)
        owners = list(self.owners)
        values[idx] = move.value
        owners[idx] = int(self.current)

        first_score = self.first_score
        second_score = self.second_score
        if self.current == Player.FIRST:
            first_score += move.value
        else:
            second_score += move.value

        return State(
            values=tuple(values),
            owners=tuple(owners),
            current=self.current.other(),
            first_score=first_score,
            second_score=second_score,
            moves_played=self.moves_played + 1,
        )

    def key(self) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        return self.values, self.owners, int(self.current)


DIRECTIONS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

LINES_4 = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
)

CORNERS = frozenset({0, BOARD_SIZE - 1, CELL_COUNT - BOARD_SIZE, CELL_COUNT - 1})


def rc_to_index(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def index_to_rc(index: int) -> tuple[int, int]:
    return divmod(index, BOARD_SIZE)


def on_board(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def is_edge_index(index: int) -> bool:
    row, col = index_to_rc(index)
    return row == 0 or row == BOARD_SIZE - 1 or col == 0 or col == BOARD_SIZE - 1


def initial_state(first_to_move: bool = True) -> State:
    values = [0] * CELL_COUNT
    owners = [EMPTY_OWNER] * CELL_COUNT

    for row, col, value in ((2, 2, 1), (3, 3, 1), (2, 3, 2), (3, 2, 2)):
        idx = rc_to_index(row, col)
        values[idx] = value
        owners[idx] = INITIAL_OWNER

    return State(
        values=tuple(values),
        owners=tuple(owners),
        current=Player.FIRST if first_to_move else Player.SECOND,
    )


def candidate_values(a: int, b: int) -> tuple[int, int, int]:
    c = a % 10
    d = b % 10
    return c + d, abs(c - d), c * d


@lru_cache(maxsize=500_000)
def legal_moves(state: State) -> tuple[Move, ...]:
    """Return legal moves after safe dominance removal.

    For each empty cell, every adjacent occupied pair extending into that cell
    generates c+d, |c-d|, and c*d from the ones digits. If multiple moves for
    the same cell have the same ones digit, only the largest value is kept.
    """

    best_by_cell_and_ones: dict[tuple[int, int], int] = {}

    for idx, owner in enumerate(state.owners):
        if owner != EMPTY_OWNER:
            continue
        row, col = index_to_rc(idx)

        for dr, dc in DIRECTIONS_8:
            r1, c1 = row + dr, col + dc
            r2, c2 = row + 2 * dr, col + 2 * dc
            if not (on_board(r1, c1) and on_board(r2, c2)):
                continue
            i1 = rc_to_index(r1, c1)
            i2 = rc_to_index(r2, c2)
            if state.owners[i1] == EMPTY_OWNER or state.owners[i2] == EMPTY_OWNER:
                continue

            for value in candidate_values(state.values[i1], state.values[i2]):
                key = (idx, value % 10)
                if value > best_by_cell_and_ones.get(key, -1):
                    best_by_cell_and_ones[key] = value

    moves = [
        Move(*index_to_rc(idx), value)
        for (idx, _ones), value in best_by_cell_and_ones.items()
    ]
    moves.sort(key=lambda mv: (mv.row, mv.col, mv.value))
    return tuple(moves)


def board_rows(state: State) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in range(BOARD_SIZE):
        rendered: list[str] = []
        for col in range(BOARD_SIZE):
            idx = rc_to_index(row, col)
            owner = state.owners[idx]
            if owner == EMPTY_OWNER:
                rendered.append(".")
            elif owner == INITIAL_OWNER:
                rendered.append(f"g{state.values[idx]}")
            elif owner == Player.FIRST:
                rendered.append(f"r{state.values[idx]}")
            else:
                rendered.append(f"b{state.values[idx]}")
        rows.append(rendered)
    return rows


def render_board(state: State) -> str:
    return "\n".join(" ".join(row) for row in board_rows(state))


def iter_empty_sandwiches(state: State) -> Iterable[tuple[int, int, int, int, int]]:
    """Yield empty-center-empty style traps as endpoint indexes and center value.

    The yielded tuples are (left_idx, center_idx, right_idx, center_value,
    center_ones). Each geometric triple is yielded once.
    """

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            center_idx = rc_to_index(row, col)
            if state.owners[center_idx] == EMPTY_OWNER:
                continue
            for dr, dc in LINES_4:
                r1, c1 = row - dr, col - dc
                r2, c2 = row + dr, col + dc
                if not (on_board(r1, c1) and on_board(r2, c2)):
                    continue
                left_idx = rc_to_index(r1, c1)
                right_idx = rc_to_index(r2, c2)
                if (
                    state.owners[left_idx] == EMPTY_OWNER
                    and state.owners[right_idx] == EMPTY_OWNER
                ):
                    value = state.values[center_idx]
                    yield left_idx, center_idx, right_idx, value, value % 10
