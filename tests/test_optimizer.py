"""Tests for the 5-pass optimizer."""

from tile_compiler import TileField, optimize
from tests.conftest import TicTacToe as TestTTT


class TestOptimize:
    def test_optimize_returns_policy(self):
        f = TileField(seed=0).train(TestTTT(), n_games=50)
        opt = optimize(f)
        assert opt.size > 0

    def test_optimize_has_stats(self):
        f = TileField(seed=0).train(TestTTT(), n_games=50)
        opt = optimize(f)
        stats = opt.stats
        assert "dead_code_removed" in stats
        assert "constants_folded" in stats
        assert "final_size" in stats

    def test_optimize_chooses(self):
        g = TestTTT()
        f = TileField(seed=0).train(g, n_games=50)
        opt = optimize(f)
        g.reset()
        action = opt.choose(g.state())
        assert isinstance(action, int)

    def test_optimize_smaller_or_equal(self):
        f = TileField(seed=0).train(TestTTT(), n_games=50)
        raw = f.export_weights()
        opt = optimize(f)
        assert opt.size <= len(raw)

    def test_repr(self):
        f = TileField(seed=0).train(TestTTT(), n_games=50)
        opt = optimize(f)
        assert "OptimizedPolicy" in repr(opt)
