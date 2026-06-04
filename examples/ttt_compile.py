"""Compile tic-tac-toe in 10 lines."""

from tile_compiler import TileField, compile

# Define a game (see conftest.py or implement the protocol)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from conftest import TicTacToe

# Train and compile — that's it
field = TileField(seed=42).train(TicTacToe(), n_games=500)
policy = compile(field)

# Use it
game = TicTacToe()
print(f"Policy covers {policy.size} states")
print(f"First move: {policy.choose(game.state())}")
