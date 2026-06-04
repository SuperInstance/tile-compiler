"""Built-in games for quickstart and testing.

These games implement the :class:`~tile_compiler.protocol.Game` protocol
so you can test the pipeline without writing your own game first.
"""

from __future__ import annotations

from typing import Any, Optional


class TicTacToe:
    """Standard 3×3 tic-tac-toe. Players are 1 (X) and 2 (O).

    State is a tuple of 9 ints (0=empty, 1=X, 2=O).
    Actions are integers 0-8 (board position).

    Example::

        from tile_compiler.games import TicTacToe
        from tile_compiler import TileField, compile

        field = TileField(seed=42).train(TicTacToe(), n_games=200)
        policy = compile(field)

        game = TicTacToe()
        while not game.is_over():
            action = policy.choose(game.state())
            if action is None:
                action = game.legal_actions()[0]
            game.step(action)
        print(f"Winner: {game.winner()}")
    """

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
        if action < 0 or action > 8:
            raise ValueError(f"Action {action} out of range [0, 8]")
        if self._board[action] != 0:
            raise ValueError(f"Cell {action} already occupied")
        if self._over:
            raise RuntimeError("Game is over — call reset() first")
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
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
            (0, 4, 8), (2, 4, 6),               # diags
        ]
        for a, b, c in lines:
            if self._board[a] == self._board[b] == self._board[c] != 0:
                self._winner = self._board[a]
                self._over = True
                return

    def __repr__(self) -> str:
        symbols = {0: ".", 1: "X", 2: "O"}
        rows = []
        for r in range(3):
            row = " ".join(symbols[self._board[r * 3 + c]] for c in range(3))
            rows.append(row)
        return "\n".join(rows)


class Connect4:
    """Standard 6×7 Connect Four. Players are 1 and 2.

    State is a tuple of 42 ints (row-major, 0=empty, 1=P1, 2=P2).
    Actions are integers 0-6 (column to drop into).

    Gravity applies — pieces fall to the lowest empty row in the column.
    """

    ROWS = 6
    COLS = 7

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._board = [0] * (self.ROWS * self.COLS)
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
        # A column is legal if the top cell is empty
        return [c for c in range(self.COLS) if self._board[c] == 0]

    def step(self, action: int) -> None:
        if action < 0 or action >= self.COLS:
            raise ValueError(f"Column {action} out of range [0, {self.COLS - 1}]")
        if self._over:
            raise RuntimeError("Game is over — call reset() first")

        # Find lowest empty row in this column
        col = action
        row = -1
        for r in range(self.ROWS - 1, -1, -1):
            idx = r * self.COLS + col
            if self._board[idx] == 0:
                row = r
                break

        if row == -1:
            raise ValueError(f"Column {action} is full")

        idx = row * self.COLS + col
        self._board[idx] = self._player
        self._check_winner(idx)
        if not self._over:
            self._player = 2 if self._player == 1 else 1
            if all(v != 0 for v in self._board):
                self._over = True

    def is_over(self) -> bool:
        return self._over

    def winner(self) -> Optional[int]:
        return self._winner

    def _check_winner(self, last_idx: int) -> None:
        """Check for four-in-a-row around the last placed piece."""
        player = self._board[last_idx]
        row = last_idx // self.COLS
        col = last_idx % self.COLS

        # Directions: horizontal, vertical, diag-right, diag-left
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1  # Include the piece just placed

            # Walk forward
            for i in range(1, 4):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < self.ROWS and 0 <= c < self.COLS:
                    if self._board[r * self.COLS + c] == player:
                        count += 1
                    else:
                        break
                else:
                    break

            # Walk backward
            for i in range(1, 4):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < self.ROWS and 0 <= c < self.COLS:
                    if self._board[r * self.COLS + c] == player:
                        count += 1
                    else:
                        break
                else:
                    break

            if count >= 4:
                self._winner = player
                self._over = True
                return

    def __repr__(self) -> str:
        symbols = {0: ".", 1: "●", 2: "○"}
        rows = []
        for r in range(self.ROWS):
            row = " ".join(symbols[self._board[r * self.COLS + c]] for c in range(self.COLS))
            rows.append(row)
        rows.append(" ".join(str(c) for c in range(self.COLS)))
        return "\n".join(rows)
