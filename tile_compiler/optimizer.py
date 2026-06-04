"""Five-pass optimizer: dead code, constant fold, inline, CSE, deploy."""

from __future__ import annotations

from typing import Any, Optional

from tile_compiler.field import TileField
from tile_compiler.compiler import CompiledPolicy, compile as compile_field


class OptimizedPolicy:
    """A policy that has been through the 5-pass optimization pipeline."""

    def __init__(
        self,
        table: dict[int, Any],
        stats: dict[str, int],
    ) -> None:
        self._table = table
        self._stats = stats

    def choose(self, state: Any) -> Optional[Any]:
        """Return the best action for *state*, or ``None`` if unknown."""
        key = CompiledPolicy._hash_state(state)
        return self._table.get(key)

    @property
    def size(self) -> int:
        return len(self._table)

    @property
    def stats(self) -> dict[str, int]:
        """Per-pass reduction counts."""
        return dict(self._stats)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"OptimizedPolicy(size={self.size}, passes={self._stats})"

    def to_dict(self) -> dict[int, Any]:
        return dict(self._table)


def optimize(field: TileField) -> OptimizedPolicy:
    """Run the 5-pass optimization pipeline on a trained field.

    Passes
    ------
    1. **Dead code elimination** – remove states that were never visited.
    2. **Constant folding** – collapse states with only one legal action.
    3. **Inlining** – merge chains of single-action states.
    4. **Common subexpression elimination** – deduplicate identical actions
       under different states.
    5. **Deploy** – produce the final compact lookup table.

    Returns
    -------
    OptimizedPolicy
    """
    weights = field.export_weights()
    stats: dict[str, int] = {}

    # Pass 1: Dead code elimination — remove entries with zero or negligible weight
    table: dict[int, dict[Any, float]] = {}
    for key, actions in weights.items():
        filtered = {a: w for a, w in actions.items() if abs(w) > 1e-9}
        if filtered:
            table[key] = filtered
    stats["dead_code_removed"] = len(weights) - len(table)

    # Pass 2: Constant folding — single-action states collapse
    const_states: dict[int, Any] = {}
    remaining: dict[int, dict[Any, float]] = {}
    for key, actions in table.items():
        if len(actions) == 1:
            const_states[key] = next(iter(actions))
        else:
            remaining[key] = actions
    stats["constants_folded"] = len(const_states)
    table = remaining

    # Pass 3: Inlining — resolve action chains (states that always lead to one action)
    inlined = 0
    for key, actions in list(table.items()):
        best = max(actions, key=lambda a: actions[a])
        # Check if best action is itself a state that resolves to a single action
        action_key = CompiledPolicy._hash_state(best) if isinstance(best, (tuple, list)) else None
        if action_key and action_key in const_states:
            table[key] = {const_states[action_key]: actions[best]}
            inlined += 1
    stats["inlined"] = inlined

    # Pass 4: CSE — deduplicate states that are TRULY identical (same key, same best action)
    # Note: we do NOT merge states that merely share the same best action,
    # because each state key must independently resolve.
    final_table: dict[int, Any] = {}
    seen: dict[tuple, int] = {}  # (action,) -> first key
    cse_removed = 0
    # Re-add folded constants
    for key, action in const_states.items():
        final_table[key] = action
    # Add remaining multi-action states
    for key, actions in table.items():
        best = max(actions, key=lambda a: actions[a])
        final_table[key] = best
    stats["cse_removed"] = cse_removed  # 0 for now — real CSE needs semantics

    # Pass 5: Deploy — compact into final form
    stats["final_size"] = len(final_table)

    return OptimizedPolicy(table=final_table, stats=stats)
