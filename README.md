# Tile Compiler — Compile Game Strategies into Fast Lookup Tables

**Tile Compiler** trains game-playing policies via tile-based Monte Carlo field training, then compiles them into zero-dependency lookup tables. It supports the full pipeline: define a game, train a tile field via self-play, compile to a `CompiledPolicy`, optimize for speed, factorize for memory efficiency, and generate JIT-compiled policies. Built-in games include Tic-Tac-Toe, Connect Four, and extensible protocol-based game definitions.

## Why It Matters

Neural network policies require runtime inference — matrix multiplications, activation functions, and memory bandwidth. Lookup table policies require only a hash and array access: O(1) with zero dependencies. For games with bounded state spaces (like Tic-Tac-Toe with ~5,478 legal positions or Connect Four with ~4.5 trillion), a compiled tile policy is both faster and more transparent than a neural network. The tile-based training approach decomposes the game state space into overlapping tiles (local patterns), learning the value of each tile independently. This decomposition enables training to converge in minutes rather than hours.

## How It Works

### Tile Field Training

A `TileField` decomposes game states into local patterns (tiles). For Tic-Tac-Toe, a tile might be "the center cell and its 4 neighbors." The field learns a value for each tile configuration via Monte Carlo simulation:

```
for each simulation game:
    for each state visited:
        extract tiles from state
        update tile values based on game outcome
```

Training is O(g × t) for g games and t tiles per state. Tile values converge to approximate minimax values.

### Compilation

`compile(field)` converts the trained tile field into a `CompiledPolicy` — a HashMap from state hash to action. The compilation step:

1. Enumerate all reachable game states (or sample if too many)
2. For each state, evaluate the tile field to get action probabilities
3. Select the best action
4. Store: hash(state) → action

Result: a pure lookup table. O(1) evaluation at runtime.

### Optimization

`optimize(policy)` applies several speedups:
- **Transposition table merging**: States that are symmetrical (rotations/reflections) share entries
- **Principal variation extraction**: Only store states on the optimal play path
- **Bit-packed keys**: Encode states as compact integers

### Factorization

`factorize(policy)` decomposes the monolithic lookup table into sub-tables by game phase (opening, midgame, endgame), enabling lazy loading.

### JIT Compilation

`jit_compile(policy)` generates a Python function with inlined decision logic — no hash lookups at all, just nested if-else statements. This is the fastest possible evaluation for small state spaces.

## Quick Start

```python
from tile_compiler import TicTacToe, TileField, CompiledPolicy

# Train a tile field
game = TicTacToe()
field = TileField()
field.train(game, num_games=5000)

# Compile to a lookup policy
policy = CompiledPolicy.from_tile_field(field)

# Use it
state = "X O  X   "
action = policy(state)
print(f"Best move: {action}")
```

```bash
pip install tile-compiler
```

## API

| Type / Function | Description |
|---|---|
| `TileField` | Trainable tile-based value field |
| `TrainingConfig` | Parameters: games, learning rate, exploration ε |
| `CompiledPolicy` | Hash → action lookup table |
| `OptimizedPolicy` | Symmetry-merged, PV-extracted |
| `FactorizedPolicy` | Phase-decomposed sub-tables |
| `JITPolicy` | Inlined if-else decision function |
| `TicTacToe`, `Connect4` | Built-in game implementations |
| `Game` protocol | Extensible interface for custom games |

## Architecture Notes

Tile compilation is the policy deployment mechanism in **SuperInstance** for bounded state spaces. Where ternary networks handle continuous state spaces, tile compilation handles discrete strategy spaces with exact lookup. The γ + η = C conservation manifests in the coverage-compression trade-off: more table entries (γ = coverage) require more memory (η = storage overhead). See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References:

- Sutton, Richard & Barto, Andrew. *Reinforcement Learning*, 2nd ed., MIT Press, 2018 — tile coding.
| Tesauro, Gerald. "Temporal Difference Learning and TD-Gammon," *CACM*, 38(3), 1995.
| Schaeffer, Jonathan et al. "Checkers Is Solved," *Science*, 317(5844), 2007 — game-solving via lookup.

## License

MIT
