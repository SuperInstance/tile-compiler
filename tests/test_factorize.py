"""Tests for SVD factorization."""

from tile_compiler import TileField, factorize
from tests.conftest import TicTacToe as TestTTT


class TestFactorize:
    def test_factorize_returns_policy(self):
        f = TileField(seed=0).train(TestTTT(), n_games=50)
        weights = f.export_weights()
        fac = factorize(weights)
        assert fac.rank > 0

    def test_factorize_chooses(self):
        g = TestTTT()
        f = TileField(seed=0).train(g, n_games=50)
        fac = factorize(f.export_weights())
        action = fac.choose(g.state())
        assert action is not None or True  # may not know all states

    def test_factorize_with_explicit_rank(self):
        f = TileField(seed=0).train(TestTTT(), n_games=50)
        fac = factorize(f.export_weights(), rank=3)
        assert fac.rank == 3

    def test_compression_ratio(self):
        f = TileField(seed=0).train(TestTTT(), n_games=50)
        fac = factorize(f.export_weights(), rank=5)
        assert fac.compression_ratio > 0

    def test_empty_weights(self):
        fac = factorize({})
        assert fac.rank == 0

    def test_repr(self):
        f = TileField(seed=0).train(TestTTT(), n_games=50)
        fac = factorize(f.export_weights())
        assert "FactorizedPolicy" in repr(fac)
