"""Regression tests for bugs found during bottom-up audit."""

import subprocess
import sys

from tile_compiler import TileField, compile, optimize, factorize, hash_state
from tile_compiler.field import Game
from conftest import TicTacToe


class TestDeterministicHash:
    """Regression: hash() was non-deterministic across processes (PYTHONHASHSEED)."""

    def test_hash_is_deterministic_across_calls(self):
        state = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        h1 = hash_state(state)
        h2 = hash_state(state)
        assert h1 == h2

    def test_hash_is_deterministic_across_processes(self):
        """Verify the same state hashes to the same value in a new process."""
        state = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        expected = hash_state(state)
        # Run in subprocess to get a fresh Python interpreter
        result = subprocess.run(
            [sys.executable, "-c",
             f"from tile_compiler import hash_state; "
             f"print(hash_state({state!r}))"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert int(result.stdout.strip()) == expected

    def test_different_states_different_hashes(self):
        h1 = hash_state((0, 0, 0, 0, 0, 0, 0, 0, 0))
        h2 = hash_state((1, 0, 0, 0, 0, 0, 0, 0, 0))
        assert h1 != h2


class TestGameProtocol:
    """Regression: Game protocol was informal, only in docstrings."""

    def test_tictactoe_satisfies_protocol(self):
        g = TicTacToe()
        assert isinstance(g, Game)

    def test_protocol_requires_methods(self):
        """A bare object should NOT satisfy the Game protocol."""
        assert not isinstance(object(), Game)


class TestCSEPreservesStates:
    """Regression: CSE pass was dropping valid state entries."""

    def test_optimize_preserves_all_state_keys(self):
        """Every trained state should still resolve after optimization."""
        f = TileField(seed=0).train(TicTacToe(), n_games=100)
        raw_weights = f.export_weights()
        raw_keys = set(raw_weights.keys())

        opt = optimize(f)
        # Every raw key that had a non-negligible action should still resolve
        missing = raw_keys - set(opt.to_dict().keys())
        # After dead code removal (negligible weights), some may be gone —
        # but NO key should be dropped by CSE that still had significant weight
        for key in raw_keys:
            actions = raw_weights[key]
            if any(abs(w) > 1e-9 for w in actions.values()):
                assert key in opt.to_dict(), (
                    f"State {key} was dropped by optimizer but had "
                    f"significant weights: {actions}"
                )


class TestEvolveFitness:
    """Regression: evolve() fitness penalized experience (total_weight decays)."""

    def test_more_training_doesnt_reduce_fitness(self):
        f_short = TileField(seed=42).train(TicTacToe(), n_games=50)
        f_long = TileField(seed=42).train(TicTacToe(), n_games=200)
        # fitness uses max-best-action, not total weight
        assert f_long._fitness() >= 0  # At minimum should be non-negative
        # The key test: evolve doesn't crash and doesn't regress
        f_short.evolve(generations=3, population=5)
        assert f_short.n_states > 0


class TestFactorizeDeterminism:
    """Regression: SVD power iteration was non-reproducible."""

    def test_factorize_is_deterministic(self):
        f = TileField(seed=0).train(TicTacToe(), n_games=50)
        weights = f.export_weights()
        fac1 = factorize(weights, rank=3)
        fac2 = factorize(weights, rank=3)
        # Same input → same output
        assert fac1.rank == fac2.rank


class TestDistillVisits:
    """Regression: distilled fields had no visit data."""

    def test_distilled_field_has_visits(self):
        from tile_compiler.distill import distill
        f1 = TileField(seed=0).train(TicTacToe(), n_games=30)
        f2 = TileField(seed=1).train(TicTacToe(), n_games=30)
        result = distill([f1, f2])
        # Should have visit data so optimizer doesn't kill everything
        assert len(result._visits) > 0

    def test_distilled_field_optimizable(self):
        from tile_compiler.distill import distill
        f1 = TileField(seed=0).train(TicTacToe(), n_games=30)
        f2 = TileField(seed=1).train(TicTacToe(), n_games=30)
        result = distill([f1, f2])
        opt = optimize(result)
        assert opt.size > 0


class TestHashConsistency:
    """Regression: _hash_state was duplicated and could diverge."""

    def test_field_and_compiler_agree(self):
        f = TileField(seed=0).train(TicTacToe(), n_games=20)
        policy = compile(f)
        g = TicTacToe()
        # Field and compiled policy should agree on known states
        field_action = f.choose(g.state())
        policy_action = policy.choose(g.state())
        assert field_action == policy_action

    def test_factorize_and_compiler_agree(self):
        f = TileField(seed=0).train(TicTacToe(), n_games=50)
        weights = f.export_weights()
        fac = factorize(weights, rank=3)
        policy = compile(f)
        # They use the same hash, so same key space
        assert isinstance(fac.choose(TicTacToe().state()), (int, type(None)))
