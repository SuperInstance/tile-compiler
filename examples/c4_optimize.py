"""Full 5-pass optimization on tic-tac-toe."""

from tile_compiler import TileField, compile, optimize
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from conftest import TicTacToe

field = TileField(seed=42).train(TicTacToe(), n_games=500)

# Raw compile vs optimized
raw = compile(field)
opt = optimize(field)

print(f"Raw policy:     {raw.size} entries")
print(f"Optimized:      {opt.size} entries")
print(f"Pass stats:     {opt.stats}")

# Both work
game = TicTacToe()
print(f"\nRaw move:   {raw.choose(game.state())}")
print(f"Opt move:   {opt.choose(game.state())}")
