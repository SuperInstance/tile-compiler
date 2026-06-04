"""tile-compiler: compile game strategies into fast lookup tables."""

import hashlib
from typing import Any

from tile_compiler.field import TileField
from tile_compiler.compiler import compile, CompiledPolicy
from tile_compiler.optimizer import optimize, OptimizedPolicy
from tile_compiler.factorize import factorize, FactorizedPolicy
from tile_compiler.jit import jit_compile, JITPolicy
from tile_compiler.hierarchical import hierarchical_compile, HierarchicalPolicy
from tile_compiler.protocol import Game, validate_game


__version__ = "0.1.0"


def hash_state(state: Any) -> int:
    """Deterministic hash for game states (blake2b, 8-byte digest).

    This is the single canonical hashing function used across
    all modules.  Every class delegates to this so that hashing
    logic never diverges.
    """
    if isinstance(state, (tuple, list)):
        data = str(tuple(state)).encode()
    else:
        data = str(state).encode()
    return int(hashlib.blake2b(data, digest_size=8).hexdigest(), 16)


__all__ = [
    "TileField",
    "compile",
    "CompiledPolicy",
    "optimize",
    "OptimizedPolicy",
    "factorize",
    "FactorizedPolicy",
    "jit_compile",
    "JITPolicy",
    "hierarchical_compile",
    "HierarchicalPolicy",
    "hash_state",
    # Protocol
    "Game",
    "validate_game",
]
