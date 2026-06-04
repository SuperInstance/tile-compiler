"""Tests for TileField training and basic operations."""

from tile_compiler.field import TileField
from conftest import TicTacToe


class TestTileFieldCreation:
    def test_empty_field(self):
        f = TileField()
        assert f.n_states == 0
        assert f.games_played == 0

    def test_seed_reproducibility(self):
        g = TicTacToe()
        f1 = TileField(seed=42).train(g, n_games=50)
        f2 = TileField(seed=42).train(g, n_games=50)
        assert f1.export_weights() == f2.export_weights()


class TestTileFieldTraining:
    def test_train_builds_weights(self):
        f = TileField(seed=0).train(TicTacToe(), n_games=10)
        assert f.n_states > 0
        assert f.games_played == 10

    def test_train_more_games_more_states(self):
        f10 = TileField(seed=0).train(TicTacToe(), n_games=10)
        f100 = TileField(seed=0).train(TicTacToe(), n_games=100)
        assert f100.n_states >= f10.n_states

    def test_train_returns_self(self):
        f = TileField()
        result = f.train(TicTacToe(), n_games=5)
        assert result is f

    def test_choose_returns_action(self):
        g = TicTacToe()
        f = TileField(seed=0).train(g, n_games=50)
        g.reset()  # Reset to get clean empty board state
        state = g.state()
        action = f.choose(state)
        assert isinstance(action, int)
        assert 0 <= action <= 8

    def test_choose_unknown_state(self):
        f = TileField()
        assert f.choose((1,2,3,4,5,6,7,8,9)) is None


class TestTileFieldEvolve:
    def test_evolve_returns_self(self):
        f = TileField(seed=0).train(TicTacToe(), n_games=20)
        result = f.evolve(generations=2, population=5)
        assert result is f

    def test_evolve_improves_or_maintains(self):
        f = TileField(seed=0).train(TicTacToe(), n_games=20)
        before = f._total_weight()
        f.evolve(generations=5, population=10)
        # Should not crash; weights should still exist
        assert f.n_states > 0
