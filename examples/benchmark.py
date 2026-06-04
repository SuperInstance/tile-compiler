"""Benchmark all compilation methods."""

import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from tile_compiler import (
    TileField, compile, optimize, factorize,
    jit_compile, hierarchical_compile,
)
from conftest import TicTacToe

N_GAMES = 500
N_QUERIES = 10_000

# Train
field = TileField(seed=42).train(TicTacToe(), n_games=N_GAMES)
weights = field.export_weights()
print(f"Trained on {N_GAMES} games → {field.n_states} states\n")

# Compile all methods
methods = {
    "Raw": compile(field),
    "Optimized": optimize(field),
    "Factorized": factorize(weights),
    "JIT": jit_compile(weights, threshold=2),
    "Hierarchical": hierarchical_compile(weights, n_clusters=8),
}

# Benchmark queries
game = TicTacToe()
states = [game.state()]  # reuse the initial state
print(f"{'Method':<15} {'Size/Info':<30} {'Queries/s':>12}")
print("-" * 60)

for name, policy in methods.items():
    # Warm up
    for _ in range(100):
        policy.choose(states[0])

    start = time.perf_counter()
    for _ in range(N_QUERIES):
        policy.choose(states[0])
    elapsed = time.perf_counter() - start
    qps = N_QUERIES / elapsed

    info = str(policy)
    print(f"{name:<15} {info:<30} {qps:>12,.0f}")
