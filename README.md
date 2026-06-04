# tile-compiler

Compile game strategies into fast lookup tables via tile-based field training.

Zero dependencies (core). Optional GPU support via `torch`.

## Quick Start

```python
from tile_compiler import TileField, compile, optimize

# Train on any game
field = TileField(seed=42)
field.train(my_game, n_games=500)

# Compile into a fast lookup table
policy = compile(field)
action = policy.choose(current_state)

# Or optimize with the 5-pass pipeline
optimized = optimize(field)
action = optimized.choose(current_state)
```

## Game Protocol

Any object with these methods works:

```python
class MyGame:
    def reset(self) -> None: ...
    def state(self) -> tuple: ...
    def legal_actions(self) -> list: ...
    def current_player(self) -> int: ...
    def step(self, action) -> None: ...
    def is_over(self) -> bool: ...
    def winner(self) -> int | None: ...
```

## Compilation Methods

### Raw Compile
```python
policy = compile(field)
# Pure hash → action lookup table. O(1) queries.
```

### 5-Pass Optimizer
```python
optimized = optimize(field)
# Pipeline: dead code → constant fold → inline → CSE → deploy
print(optimized.stats)  # Per-pass reduction counts
```

### SVD Factorization
```python
from tile_compiler import factorize
fac = factorize(field.export_weights(), rank=5)
print(f"Compression: {fac.compression_ratio:.1f}x")
```

### JIT Compilation
```python
from tile_compiler import jit_compile
jit = jit_compile(field.export_weights(), threshold=3)
# Compiles hot paths on-the-fly after N visits
print(jit.hit_rate)  # Cache effectiveness
```

### Hierarchical (k-means)
```python
from tile_compiler import hierarchical_compile
h = hierarchical_compile(field.export_weights(), n_clusters=8)
# 2-level lookup: state → cluster → action
```

## Benchmarks

On tic-tac-toe (500 training games, 10K queries):

| Method | Lookup Speed | Notes |
|--------|-------------|-------|
| Raw | ~1M qps | Full table |
| Optimized | ~1M qps | Smaller table, same speed |
| Factorized | ~50K qps | Compressed, slower reconstruction |
| JIT | ~500K qps (warm) | Hot-path caching |
| Hierarchical | ~800K qps | Clustered lookup |

## Installation

```bash
pip install -e .

# With GPU support
pip install -e ".[gpu]"

# Dev dependencies
pip install -e ".[dev]"
pytest
```

## Project Structure

```
tile_compiler/
├── field.py        # TileField — train, evolve, choose
├── compiler.py     # compile() → CompiledPolicy
├── optimizer.py    # 5-pass pipeline
├── factorize.py    # SVD compression
├── jit.py          # Hot-path JIT compilation
├── hierarchical.py # k-means meta-tiles
├── distill.py      # Ensemble distillation
└── gpu.py          # Optional GPU batch ops
```

## License

MIT
