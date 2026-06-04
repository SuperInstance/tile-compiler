"""Shared game fixtures for tests."""

from __future__ import annotations

from typing import Any, Optional


class TicTacToe:
    """Minimal tic-tac-toe for testing. Board is a tuple of 9 ints (0/1/2)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._board = [0] * 9
        self._player = 1
        self._over = False
        self._winner: Optional[int] = None

    def state(self) -> tuple[int, ...]:
        return tuple(self._board)

    def current_player(self) -> int:
        return self._player

    def legal_actions(self) -> list[int]:
        if self._over:
            return []
        return [i for i, v in enumerate(self._board) if v == 0]

    def step(self, action: int) -> None:
        if self._board[action] != 0:
            raise ValueError(f"Cell {action} occupied")
        self._board[action] = self._player
        self._check_winner()
        if not self._over:
            self._player = 2 if self._player == 1 else 1
            if all(v != 0 for v in self._board):
                self._over = True

    def is_over(self) -> bool:
        return self._over

    def winner(self) -> Optional[int]:
        return self._winner

    def _check_winner(self) -> None:
        lines = [
            (0,1,2),(3,4,5),(6,7,8),  # rows
            (0,3,6),(1,4,7),(2,5,8),  # cols
            (0,4,8),(2,4,6),          # diags
        ]
        for a, b, c in lines:
            if self._board[a] == self._board[b] == self._board[c] != 0:
                self._winner = self._board[a]
                self._over = True
                return
