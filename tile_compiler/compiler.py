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
        # hash_fn is kept for backward compatibility but ignored;
        # _hash_state (blake2b) is always used.
        self._hash_fn = hash_fn

    def choose(self, state: Any) -> Optional[Any]:
        """Return the best action for *state*, or ``None`` if unknown."""
        key = self._hash_state(state)
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
        """Deterministic hash — delegates to canonical ``hash_state``."""
        from tile_compiler import hash_state
        return hash_state(state)

    def to_dict(self) -> dict[int, Any]:
        """Return the raw lookup table."""
        return dict(self._table)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save compiled policy to a JSON file.

        The saved file can be loaded on any platform with zero
        dependencies — no numpy, no torch, no tile_compiler needed
        for the raw lookup.
        """
        import json
        from pathlib import Path

        data = {
            "version": 1,
            "size": self.size,
            "table": {str(k): v for k, v in self._table.items()},
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "CompiledPolicy":
        """Load a compiled policy from a JSON file."""
        import json
        from pathlib import Path

        data = json.loads(Path(path).read_text())
        table = {int(k): v for k, v in data["table"].items()}
        return cls(table=table)

    def to_python(self, function_name: str = "choose") -> str:
        """Export as a standalone Python function with zero external dependencies.

        Uses only hashlib (stdlib) + a dict lookup. No numpy, torch, or
        tile_compiler imports needed. Suitable for deployment on any
        Python-capable system including MicroPython and embedded devices.
        """
        lines = [
            'import hashlib',
            '',
            '',
            f'def {function_name}(state):',
            '    """Compiled tile policy — zero external dependencies."""',
            '    _table = {',
        ]
        for key, action in self._table.items():
            action_repr = repr(action)
            lines.append(f'        {key}: {action_repr},')
        lines.extend([
            '    }',
            '    if isinstance(state, (tuple, list)):',
            '        data = str(tuple(state)).encode()',
            '    else:',
            '        data = str(state).encode()',
            '    key = int(hashlib.blake2b(data, digest_size=8).hexdigest(), 16)',
            '    return _table.get(key)',
            '',
        ])
        return "\n".join(lines)


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


# Alias that avoids shadowing Python's builtin compile()
compile_field = compile
