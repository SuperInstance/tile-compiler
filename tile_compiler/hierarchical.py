"""Hierarchical compilation: k-means meta-tiles, 2-level lookup."""

from __future__ import annotations

import math
import random
from typing import Any, Optional

from tile_compiler.compiler import CompiledPolicy


class HierarchicalPolicy:
    """Two-level hierarchical policy using clustered meta-tiles.

    Level 1 maps a state to a cluster (meta-tile).
    Level 2 maps a (cluster, action) pair to the best action.

    This reduces table size when many states share similar behaviour.
    """

    def __init__(
        self,
        centroids: list[list[float]],
        assignments: dict[int, int],
        cluster_tables: list[dict[int, Any]],
        state_keys: list[int],
        centroid_actions: Optional[dict[int, Any]] = None,
        action_list: Optional[list[Any]] = None,
    ) -> None:
        self._centroids = centroids
        self._assignments = assignments
        self._cluster_tables = cluster_tables
        self._state_keys = state_keys
        self._centroid_action = centroid_actions or {}
        self._action_list = action_list or []

    def choose(self, state: Any) -> Optional[Any]:
        key = CompiledPolicy._hash_state(state)
        cluster_id = self._assignments.get(key)
        if cluster_id is not None:
            table = self._cluster_tables[cluster_id]
            result = table.get(key)
            if result is not None:
                return result

        # Centroid fallback: convert state to feature vector and find nearest
        try:
            vec = self._state_to_vector(state)
            nearest = self._nearest_centroid(vec)
            return self._centroid_action.get(nearest)
        except (TypeError, ValueError):
            return None

    @property
    def n_clusters(self) -> int:
        return len(self._centroids)

    @property
    def total_entries(self) -> int:
        return sum(len(t) for t in self._cluster_tables)

    def _state_to_vector(self, state: Any) -> list[float]:
        """Convert a state to a feature vector for centroid matching."""
        # Use blake2b digest bytes as feature vector (deterministic, fast)
        import hashlib
        if isinstance(state, (tuple, list)):
            data = str(tuple(state)).encode()
        else:
            data = str(state).encode()
        digest = hashlib.blake2b(data, digest_size=8).digest()
        return [float(b) / 255.0 for b in digest]

    def _nearest_centroid(self, vec: list[float]) -> int:
        """Find the nearest centroid by Euclidean distance."""
        best_cluster = 0
        best_dist = float("inf")
        for c, cent in enumerate(self._centroids):
            # Ensure centroid and vector have compatible dimensions
            dim = min(len(cent), len(vec))
            if dim == 0:
                continue
            dist = sum((vec[i] - cent[i]) ** 2 for i in range(dim))
            if dist < best_dist:
                best_dist = dist
                best_cluster = c
        return best_cluster

    def __repr__(self) -> str:
        return (
            f"HierarchicalPolicy("
            f"clusters={self.n_clusters}, "
            f"entries={self.total_entries})"
        )


def hierarchical_compile(
    field_weights: dict[int, dict[Any, float]],
    n_clusters: int = 8,
    max_iterations: int = 50,
) -> HierarchicalPolicy:
    """Compile weights into a 2-level hierarchical policy using k-means.

    Parameters
    ----------
    field_weights:
        The raw weight table.
    n_clusters:
        Number of meta-tiles (clusters).
    max_iterations:
        Max k-means iterations.

    Returns
    -------
    HierarchicalPolicy
    """
    if not field_weights:
        return HierarchicalPolicy([], {}, [], [])

    state_keys = list(field_weights.keys())
    action_set: set[Any] = set()
    for actions in field_weights.values():
        action_set.update(actions.keys())
    action_list = sorted(action_set, key=str)
    action_map = {a: i for i, a in enumerate(action_list)}
    dim = len(action_list)

    # Build feature vectors
    vectors: list[list[float]] = []
    for key in state_keys:
        actions = field_weights[key]
        vec = [actions.get(a, 0.0) for a in action_list]
        vectors.append(vec)

    n = len(vectors)
    k = min(n_clusters, n)

    # k-means clustering
    rng = random.Random(42)
    indices = rng.sample(range(n), k)
    centroids = [vectors[i][:] for i in indices]

    assignments = [0] * n
    for _ in range(max_iterations):
        changed = False
        # Assign
        for i, vec in enumerate(vectors):
            best_cluster = 0
            best_dist = float("inf")
            for c, cent in enumerate(centroids):
                dist = sum((a - b) ** 2 for a, b in zip(vec, cent))
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = c
            if assignments[i] != best_cluster:
                assignments[i] = best_cluster
                changed = True

        if not changed:
            break

        # Update centroids
        for c in range(k):
            members = [i for i in range(n) if assignments[i] == c]
            if members:
                centroids[c] = [
                    sum(vectors[i][j] for i in members) / len(members)
                    for j in range(dim)
                ]

    # Build per-cluster tables
    assignment_map: dict[int, int] = {}
    cluster_tables: list[dict[int, Any]] = [{} for _ in range(k)]

    for i, key in enumerate(state_keys):
        c = assignments[i]
        assignment_map[key] = c
        actions = field_weights[key]
        if actions:
            cluster_tables[c][key] = max(actions, key=lambda a: actions[a])

    # Compute centroid actions (majority vote per cluster)
    centroid_actions: dict[int, Any] = {}
    for c in range(k):
        action_counts: dict[Any, int] = {}
        for table in [cluster_tables[c]]:
            for key, action in table.items():
                action_counts[action] = action_counts.get(action, 0) + 1
        if action_counts:
            centroid_actions[c] = max(action_counts, key=lambda a: action_counts[a])

    return HierarchicalPolicy(
        centroids=centroids,
        assignments=assignment_map,
        cluster_tables=cluster_tables,
        state_keys=state_keys,
        centroid_actions=centroid_actions,
        action_list=action_list,
    )
