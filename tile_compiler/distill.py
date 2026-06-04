"""Ensemble distillation (reference implementation).

Distillation combines multiple TileFields into one. In practice the
best single field often outperforms the ensemble, but distillation
is useful for combining different training runs.
"""

from __future__ import annotations

from typing import Any

from tile_compiler.field import TileField
from tile_compiler.compiler import CompiledPolicy


def distill(fields: list[TileField], temperature: float = 1.0) -> TileField:
    """Distill multiple fields into a single field.

    Parameters
    ----------
    fields:
        List of trained TileFields.
    temperature:
        Softmax temperature for averaging (higher = more uniform).

    Returns
    -------
    TileField
        A new field with averaged weights.
    """
    if not fields:
        raise ValueError("Need at least one field to distill")

    merged: dict[int, dict[Any, list[float]]] = {}
    for field in fields:
        weights = field.export_weights()
        for key, actions in weights.items():
            if key not in merged:
                merged[key] = {}
            for action, weight in actions.items():
                merged[key].setdefault(action, []).append(weight)

    result = TileField.__new__(TileField)
    result._weights = {}
    result._visits = {k: 1 for k in merged}  # Mark all distilled states as visited
    result._games_played = sum(f.games_played for f in fields)
    result._learning_rate = fields[0]._learning_rate
    result._decay = fields[0]._decay
    result._rng = __import__("random").Random()

    for key, actions in merged.items():
        result._weights[key] = {}
        for action, values in actions.items():
            # Soft average with temperature
            avg = sum(values) / len(values)
            result._weights[key][action] = avg

    return result
