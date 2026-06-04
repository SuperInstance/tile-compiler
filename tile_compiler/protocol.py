"""Game and Policy protocols: the interfaces tile-compiler expects."""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


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


@runtime_checkable
class Policy(Protocol):
    """Unified interface for all compiled/optimized policies.

    Every policy type (CompiledPolicy, OptimizedPolicy, etc.) must
    implement this interface. This prevents hash/choose drift across
    policy implementations — the bug the builder found in Cycle 1.
    """

    def choose(self, state: Any) -> Optional[Any]:
        """Return the best action for a game state, or None if unknown."""
        ...

    @property
    def size(self) -> int:
        """Number of entries in the policy lookup."""
        ...


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
