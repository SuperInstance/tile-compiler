"""Game protocol: the interface tile-compiler expects from games."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Game(Protocol):
    """The game protocol that tile-compiler expects.

    Any object with these methods works as a game. Use this protocol
    for type checking::

        from tile_compiler.protocol import Game

        class MyGame:
            def reset(self) -> None: ...
            def state(self) -> tuple: ...
            def legal_actions(self) -> list: ...
            def step(self, action) -> None: ...
            def is_over(self) -> bool: ...
            def winner(self) -> int | None: ...
            def current_player(self) -> int: ...

        field.train(MyGame())  # type-checked
    """

    def reset(self) -> None: ...
    def state(self) -> Any: ...
    def legal_actions(self) -> list: ...
    def step(self, action: Any) -> None: ...
    def is_over(self) -> bool: ...
    def winner(self) -> Any | None: ...


def validate_game(game: Any) -> list[str]:
    """Validate that a game object conforms to the Game protocol.

    Returns a list of issues. Empty list = valid.
    """
    issues: list[str] = []
    required = ["reset", "state", "legal_actions", "step", "is_over", "winner"]
    for method in required:
        if not hasattr(game, method):
            issues.append(f"Missing method: {method}()")
        elif not callable(getattr(game, method)):
            issues.append(f"{method} exists but is not callable")
    return issues
