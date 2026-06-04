"""SVD-based factorization and rank-r compression for weight matrices."""

from __future__ import annotations

import math
import random
from typing import Any, Optional


class FactorizedPolicy:
    """A policy using SVD-compressed weight matrices."""

    def __init__(
        self,
        u: list[list[float]],
        s: list[float],
        vt: list[list[float]],
        action_index: list[Any],
        state_keys: list[int],
        rank: int,
    ) -> None:
        self._u = u
        self._s = s
        self._vt = vt
        self._action_index = action_index
        self._state_keys = state_keys
        self._rank = rank
        self._table = self._reconstruct()

    def choose(self, state: Any) -> Optional[Any]:
        key = self._hash_state(state)
        return self._table.get(key)

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def compression_ratio(self) -> float:
        """Original entries vs compressed (rank * (m + n))."""
        m, n = len(self._state_keys), len(self._action_index)
        original = m * n
        compressed = self._rank * (m + n)
        return original / compressed if compressed > 0 else 1.0

    def __repr__(self) -> str:
        return (
            f"FactorizedPolicy(rank={self._rank}, "
            f"states={len(self._state_keys)}, "
            f"actions={len(self._action_index)}, "
            f"compression={self.compression_ratio:.1f}x)"
        )

    def _reconstruct(self) -> dict[int, Any]:
        """Reconstruct the lookup table from SVD components."""
        table: dict[int, Any] = {}
        for i, state_key in enumerate(self._state_keys):
            # Find best action: argmax of reconstructed row
            best_action = self._action_index[0]
            best_score = float("-inf")
            for j, action in enumerate(self._action_index):
                score = sum(
                    self._u[i][k] * self._s[k] * self._vt[k][j]
                    for k in range(self._rank)
                )
                if score > best_score:
                    best_score = score
                    best_action = action
            table[state_key] = best_action
        return table

    @staticmethod
    def _hash_state(state: Any) -> int:
        from tile_compiler import hash_state
        return hash_state(state)


def factorize(
    field_weights: dict[int, dict[Any, float]],
    rank: Optional[int] = None,
    energy: float = 0.95,
) -> FactorizedPolicy:
    """Factorize a weight table using truncated SVD.

    Parameters
    ----------
    field_weights:
        The raw weight table (state_key -> {action: weight}).
    rank:
        Explicit rank for truncation. If ``None``, choose rank to
        capture *energy* fraction of total singular value energy.
    energy:
        Fraction of singular value energy to preserve (default 0.95).

    Returns
    -------
    FactorizedPolicy
    """
    if not field_weights:
        return FactorizedPolicy([], [], [], [], [], 0)

    # Build action index and state list
    action_set: set[Any] = set()
    for actions in field_weights.values():
        action_set.update(actions.keys())
    action_index = sorted(action_set, key=str)
    state_keys = list(field_weights.keys())
    action_map = {a: j for j, a in enumerate(action_index)}

    m, n = len(state_keys), len(action_index)

    # Build weight matrix
    matrix = [[0.0] * n for _ in range(m)]
    for i, key in enumerate(state_keys):
        for action, weight in field_weights[key].items():
            matrix[i][action_map[action]] = weight

    # SVD via power iteration (zero-dependency, deterministic seed)
    u, s, vt = _svd_power_iteration(matrix, rank=min(m, n))

    # Choose rank
    if rank is None:
        total_energy = sum(si * si for si in s)
        cumulative = 0.0
        rank = len(s)
        for k, si in enumerate(s):
            cumulative += si * si
            if total_energy > 0 and cumulative / total_energy >= energy:
                rank = k + 1
                break

    # Truncate
    u = [row[:rank] for row in u]  # m x rank
    s = s[:rank]                     # rank
    # vt is rank x n (each row is a right singular vector)
    vt_truncated = vt[:rank]        # rank x n

    return FactorizedPolicy(
        u=[row[:rank] for row in u],
        s=s[:rank],
        vt=vt_truncated[:rank] if len(vt_truncated) >= rank else vt_truncated,
        action_index=action_index,
        state_keys=state_keys,
        rank=rank,
    )


def _svd_power_iteration(
    matrix: list[list[float]], rank: int, iterations: int = 100, seed: int = 42
) -> tuple[list[list[float]], list[float], list[list[float]]]:
    """Compute truncated SVD using power iteration. Zero dependencies."""
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    if m == 0 or n == 0:
        return [], [], []

    rank = min(rank, m, n)
    rng = random.Random(seed)
    u_vectors: list[list[float]] = []
    singular_values: list[float] = []
    v_vectors: list[list[float]] = []

    # Residual matrix
    residual = [row[:] for row in matrix]

    for _ in range(rank):
        v = [rng.gauss(0, 1) for _ in range(n)]

        for _ in range(iterations):
            # u = residual @ v
            u = [sum(residual[i][j] * v[j] for j in range(n)) for i in range(m)]
            # Normalize u
            norm_u = math.sqrt(sum(x * x for x in u))
            if norm_u < 1e-12:
                break
            u = [x / norm_u for x in u]

            # v = residual^T @ u
            v = [sum(residual[i][j] * u[i] for i in range(m)) for j in range(n)]
            norm_v = math.sqrt(sum(x * x for x in v))
            if norm_v < 1e-12:
                break
            v = [x / norm_v for x in v]

        sigma = sum(
            residual[i][j] * u[i] * v[j]
            for i in range(m)
            for j in range(n)
        )
        sigma = abs(sigma)

        u_vectors.append(u)
        singular_values.append(sigma)
        v_vectors.append(v)

        # Deflate
        for i in range(m):
            for j in range(n):
                residual[i][j] -= sigma * u[i] * v[j]

    # Format: u as m x rank matrix, s as list, vt as rank x n matrix
    u_matrix = [[u_vectors[k][i] for k in range(rank)] for i in range(m)]
    vt_matrix = v_vectors[:rank]  # rank x n

    return u_matrix, singular_values, vt_matrix
