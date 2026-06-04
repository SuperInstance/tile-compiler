"""tile-compiler: compile game strategies into fast lookup tables."""

from tile_compiler.field import TileField
from tile_compiler.compiler import compile, CompiledPolicy
from tile_compiler.optimizer import optimize, OptimizedPolicy
from tile_compiler.factorize import factorize, FactorizedPolicy
from tile_compiler.jit import jit_compile, JITPolicy
from tile_compiler.hierarchical import hierarchical_compile, HierarchicalPolicy

__version__ = "0.1.0"
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
]
