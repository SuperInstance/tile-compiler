"""Tests for v0.3 additions: TrainingConfig, Policy protocol."""

import pytest
from tile_compiler.config import TrainingConfig, DEFAULT_CONFIG
from tile_compiler.protocol import Game, Policy, validate_game
from tile_compiler import TileField, compile_field, optimize
from tile_compiler.games import TicTacToe


class TestTrainingConfig:
    """TrainingConfig: immutable config prevents parameter drift."""

    def test_default_values(self):
        c = TrainingConfig()
        assert c.temperature == 0.3
        assert c.score_decay == 0.005
        assert c.weight_decay == 0.95
        assert c.learning_rate == 0.1
        assert c.explore_rate == 0.3

    def test_custom_values(self):
        c = TrainingConfig(temperature=0.5, score_decay=0.01)
        assert c.temperature == 0.5
        assert c.score_decay == 0.01
        assert c.weight_decay == 0.95  # Unchanged

    def test_frozen(self):
        c = TrainingConfig()
        with pytest.raises(AttributeError):
            c.temperature = 0.9  # type: ignore

    def test_with_overrides(self):
        c = TrainingConfig(temperature=0.3)
        c2 = c.with_overrides(temperature=0.5)
        assert c2.temperature == 0.5
        assert c.temperature == 0.3  # Original unchanged

    def test_with_overrides_multiple(self):
        c = TrainingConfig()
        c2 = c.with_overrides(temperature=0.1, explore_rate=0.1, score_decay=0.01)
        assert c2.temperature == 0.1
        assert c2.explore_rate == 0.1
        assert c2.score_decay == 0.01

    def test_default_config_singleton(self):
        assert DEFAULT_CONFIG.temperature == 0.3
        assert DEFAULT_CONFIG.score_decay == 0.005

    def test_repr(self):
        c = TrainingConfig(temperature=0.5)
        r = repr(c)
        assert "T=0.5" in r
        assert "decay=0.005" in r

    def test_weight_decay_not_score_decay(self):
        """The builder's bug: these are separate concepts."""
        c = TrainingConfig()
        assert c.weight_decay != c.score_decay
        assert c.weight_decay == 0.95   # Per-update retention
        assert c.score_decay == 0.005   # Per-game memory fade


class TestPolicyProtocol:
    """Policy protocol: unified interface for all policy types."""

    def test_compiled_policy_is_policy(self):
        game = TicTacToe()
        field = TileField(seed=42).train(game, n_games=50)
        policy = compile_field(field)
        assert isinstance(policy, Policy)

    def test_optimized_policy_is_policy(self):
        game = TicTacToe()
        field = TileField(seed=42).train(game, n_games=50)
        policy = optimize(field)
        assert isinstance(policy, Policy)

    def test_policy_has_choose(self):
        game = TicTacToe()
        field = TileField(seed=42).train(game, n_games=50)
        policy = compile_field(field)
        assert hasattr(policy, "choose")
        assert callable(policy.choose)

    def test_policy_has_size(self):
        game = TicTacToe()
        field = TileField(seed=42).train(game, n_games=50)
        policy = compile_field(field)
        assert hasattr(policy, "size")
        assert policy.size > 0

    def test_all_policy_types_have_same_hash(self):
        """All policy types must use the same hash function."""
        from tile_compiler import hash_state
        game = TicTacToe()
        game.reset()
        state = game.state()
        expected = hash_state(state)

        field = TileField(seed=42).train(game, n_games=50)
        compiled = compile_field(field)

        # CompiledPolicy must use hash_state internally
        assert compiled._hash_state(state) == expected
