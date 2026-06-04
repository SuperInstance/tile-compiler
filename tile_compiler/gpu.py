"""Optional GPU-accelerated batch operations (requires torch)."""

from __future__ import annotations

from typing import Any, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def gpu_batch_evaluate(
    field_weights: dict[int, dict[Any, float]],
    states: list[Any],
    device: str = "cpu",
) -> list[Optional[Any]]:
    """Evaluate many states at once on GPU.

    Parameters
    ----------
    field_weights:
        The weight table.
    states:
        States to evaluate.
    device:
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    list of best actions (None for unknown states).

    Raises
    ------
    ImportError
        If torch is not installed.
    """
    if not HAS_TORCH:
        raise ImportError(
            "GPU operations require torch. "
            "Install with: pip install tile-compiler[gpu]"
        )

    if not field_weights or not states:
        return [None] * len(states)

    # Build action index
    action_set: set[Any] = set()
    for actions in field_weights.values():
        action_set.update(actions.keys())
    action_list = sorted(action_set, key=str)
    action_map = {a: i for i, a in enumerate(action_list)}
    state_keys = list(field_weights.keys())
    state_map = {k: i for i, k in enumerate(state_keys)}
    n_states = len(state_keys)
    n_actions = len(action_list)

    # Build weight tensor
    weight_tensor = torch.zeros(n_states, n_actions, device=device)
    for key, actions in field_weights.items():
        si = state_map[key]
        for action, weight in actions.items():
            weight_tensor[si, action_map[action]] = weight

    # Build state lookup tensor
    results: list[Optional[Any]] = []
    for state in states:
        key = hash(tuple(state)) if isinstance(state, (tuple, list)) else hash(state)
        idx = state_map.get(key)
        if idx is None:
            results.append(None)
        else:
            best_idx = torch.argmax(weight_tensor[idx]).item()
            results.append(action_list[best_idx])

    return results


def gpu_available() -> bool:
    """Check if GPU operations are available."""
    if not HAS_TORCH:
        return False
    return torch.cuda.is_available()
