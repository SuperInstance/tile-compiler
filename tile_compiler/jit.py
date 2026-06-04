"""JIT compilation: threshold-based hot-path discovery and caching."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from tile_compiler.compiler import CompiledPolicy


class JITPolicy:
    """Just-in-time compiled policy with hot-path caching.

    Unlike :class:`CompiledPolicy`, which pre-compiles everything,
    JITPolicy starts with an interpreter and compiles only the hot
    paths that exceed a visit threshold.
    """

    def __init__(
        self,
        interpreter_fn: Callable[[Any], Optional[Any]],
        threshold: int = 3,
    ) -> None:
        self._interpreter = interpreter_fn
        self._threshold = threshold
        self._cache: dict[int, Any] = {}
        self._visit_counts: dict[int, int] = {}
        self._compiled: set[int] = set()
        self._hits = 0
        self._misses = 0

    def choose(self, state: Any) -> Optional[Any]:
        """Choose action, compiling hot paths on the fly."""
        key = CompiledPolicy._hash_state(state)

        # Fast path: already compiled
        if key in self._compiled:
            self._hits += 1
            return self._cache[key]

        # Count visit
        self._visit_counts[key] = self._visit_counts.get(key, 0) + 1
        self._misses += 1

        # If hot enough, compile and cache
        if self._visit_counts[key] >= self._threshold:
            result = self._interpreter(state)
            self._cache[key] = result
            self._compiled.add(key)
            return result

        # Cold path: interpret
        return self._interpreter(state)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "compiled_paths": len(self._compiled),
            "total_visits": sum(self._visit_counts.values()),
            "cache_hits": self._hits,
            "cache_misses": self._misses,
        }

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"JITPolicy(compiled={len(self._compiled)}, "
            f"hit_rate={self.hit_rate:.1%})"
        )


def jit_compile(
    field_weights: dict[int, dict[Any, float]],
    threshold: int = 3,
) -> JITPolicy:
    """Create a JIT-compiled policy from a weight table.

    Parameters
    ----------
    field_weights:
        The raw weight table.
    threshold:
        Number of visits before a state path is compiled.

    Returns
    -------
    JITPolicy
    """
    weights = {
        k: dict(v) for k, v in field_weights.items()
    }

    def interpreter(state: Any) -> Optional[Any]:
        key = CompiledPolicy._hash_state(state)
        actions = weights.get(key)
        if not actions:
            return None
        return max(actions, key=lambda a: actions[a])

    return JITPolicy(interpreter, threshold=threshold)
