"""Tests for JIT compilation."""

from tile_compiler import TileField, jit_compile
from conftest import TicTacToe


class TestJIT:
    def test_jit_creates_policy(self):
        f = TileField(seed=0).train(TicTacToe(), n_games=50)
        jit = jit_compile(f.export_weights())
        assert jit.stats["compiled_paths"] == 0

    def test_jit_compiles_hot_paths(self):
        f = TileField(seed=0).train(TicTacToe(), n_games=50)
        jit = jit_compile(f.export_weights(), threshold=2)
        g = TicTacToe()
        state = g.state()
        # Call multiple times to trigger compilation
        jit.choose(state)
        jit.choose(state)
        jit.choose(state)
        assert jit.stats["compiled_paths"] > 0

    def test_jit_hit_rate(self):
        f = TileField(seed=0).train(TicTacToe(), n_games=50)
        jit = jit_compile(f.export_weights(), threshold=2)
        g = TicTacToe()
        state = g.state()
        for _ in range(10):
            jit.choose(state)
        assert jit.hit_rate > 0

    def test_jit_unknown_state(self):
        f = TileField(seed=0).train(TicTacToe(), n_games=50)
        jit = jit_compile(f.export_weights())
        assert jit.choose((9,) * 9) is None

    def test_repr(self):
        f = TileField(seed=0).train(TicTacToe(), n_games=50)
        jit = jit_compile(f.export_weights())
        assert "JITPolicy" in repr(jit)
