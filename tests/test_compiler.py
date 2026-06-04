"""Tests for compile()."""

from tile_compiler import TileField, compile, CompiledPolicy
from conftest import TicTacToe


class TestCompile:
    def test_compile_returns_policy(self):
        f = TileField(seed=0).train(TicTacToe(), n_games=20)
        policy = compile(f)
        assert isinstance(policy, CompiledPolicy)

    def test_compiled_policy_chooses(self):
        g = TicTacToe()
        f = TileField(seed=0).train(g, n_games=50)
        policy = f.compile()
        g.reset()
        action = policy.choose(g.state())
        assert isinstance(action, int)
        assert 0 <= action <= 8

    def test_compiled_policy_size(self):
        f = TileField(seed=0).train(TicTacToe(), n_games=20)
        policy = compile(f)
        assert policy.size > 0
        assert len(policy) == policy.size

    def test_unknown_state_returns_none(self):
        f = TileField(seed=0).train(TicTacToe(), n_games=20)
        policy = compile(f)
        assert policy.choose((9,) * 9) is None

    def test_contains(self):
        g = TicTacToe()
        f = TileField(seed=0).train(g, n_games=20)
        policy = compile(f)
        # The initial empty board should be known
        g.reset()
        assert g.state() in policy

    def test_to_dict(self):
        f = TileField(seed=0).train(TicTacToe(), n_games=20)
        policy = compile(f)
        d = policy.to_dict()
        assert isinstance(d, dict)
        assert len(d) == policy.size

    def test_repr(self):
        f = TileField(seed=0).train(TicTacToe(), n_games=20)
        policy = compile(f)
        r = repr(policy)
        assert "CompiledPolicy" in r
