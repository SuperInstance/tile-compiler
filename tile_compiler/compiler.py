"""Compile a TileField into a zero-dependency lookup-table policy."""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tile_compiler.field import TileField


class CompiledPolicy:
    """A fast, read-only policy backed by a lookup table.

    Use :func:`compile` to create instances.
    """

    def __init__(self, table: dict[int, Any], hash_fn: Any = None) -> None:
        self._table = table
        self._hash_fn = hash_fn or hash

    def choose(self, state: Any) -> Optional[Any]:
        """Return the best action for *state*, or ``None`` if unknown."""
        key = self._hash_fn(state) if self._hash_fn is not hash else self._hash_state(state)
        return self._table.get(key)

    @property
    def size(self) -> int:
        """Number of entries in the lookup table."""
        return len(self._table)

    def __contains__(self, state: Any) -> bool:
        key = self._hash_state(state)
        return key in self._table

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"CompiledPolicy(size={self.size})"

    @staticmethod
    def _hash_state(state: Any) -> int:
        if isinstance(state, (tuple, list)):
            return hash(tuple(state))
        return hash(state)

    def to_dict(self) -> dict[int, Any]:
        """Return the raw lookup table."""
        return dict(self._table)


def compile(field: TileField) -> CompiledPolicy:
    """Compile a :class:`TileField` into a :class:`CompiledPolicy`.

    The compiled policy is a pure lookup table: for every state seen
    during training, store the action with the highest weight.

    Parameters
    ----------
    field:
        A trained :class:`TileField`.

    Returns
    -------
    CompiledPolicy
    """
    weights = field.export_weights()
    table: dict[int, Any] = {}
    for state_key, actions in weights.items():
        if actions:
            table[state_key] = max(actions, key=lambda a: actions[a])
    return CompiledPolicy(table)
