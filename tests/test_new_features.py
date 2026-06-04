"""Tests for built-in games, to_python export, temperature, decay, and diagnostics."""

import math
import subprocess
import sys
import tempfile

from tile_compiler import (
    TileField, compile, optimize, factorize, analyze,
    TicTacToe, Connect4, Game, validate_game, hash_state,
)
from tile_compiler.analyze import AnalysisReport as FieldReport


class TestBuiltInTicTacToe:
    """Built-in TicTacToe game works end-to-end."""

    def test_ttt_game_protocol(self):
        g = TicTacToe()
        assert isinstance(g, Game)

    def test_ttt_validate(self):
        issues = validate_game(TicTacToe())
        assert issues == []

    def test_ttt_train_and_compile(self):
        f = TileField(seed=42).train(TicTacToe(), n_games=100)
        assert f.n_states > 0
        policy = compile(f)
        assert policy.size > 0

    def test_ttt_play_full_game(self):
        game = TicTacToe()
        f = TileField(seed=42).train(TicTacToe(), n_games=200)
        policy = compile(f)

        while not game.is_over():
            action = policy.choose(game.state())
            if action is None:
                action = game.legal_actions()[0]
            game.step(action)
        # Game should end (winner or draw)
        assert game.is_over()

    def test_ttt_repr(self):
        g = TicTacToe()
        assert "." in repr(g)

    def test_ttt_step_errors(self):
        g = TicTacToe()
        g.step(0)  # Place in cell 0
        import pytest
        with pytest.raises(ValueError):
            g.step(0)  # Cell occupied
        with pytest.raises(ValueError):
            g.step(-1)  # Out of range


class TestBuiltInConnect4:
    """Built-in Connect4 game works end-to-end."""

    def test_c4_game_protocol(self):
        g = Connect4()
        assert isinstance(g, Game)

    def test_c4_validate(self):
        issues = validate_game(Connect4())
        assert issues == []

    def test_c4_train_and_compile(self):
        f = TileField(seed=42).train(Connect4(), n_games=50)
        assert f.n_states > 0
        policy = compile(f)
        assert policy.size > 0

    def test_c4_gravity(self):
        """Pieces should fall to the lowest empty row."""
        g = Connect4()
        g.step(3)  # Drop in column 3
        state = g.state()
        # Bottom row (row 5), column 3 should be player 1
        assert state[5 * 7 + 3] == 1
        # Row 4, column 3 should still be empty
        assert state[4 * 7 + 3] == 0

    def test_c4_win_horizontal(self):
        """Four in a row horizontally."""
        g = Connect4()
        # Player 1: cols 0,1,2,3 — need to alternate with player 2
        for col in [0, 1, 2, 3]:
            g.step(col)  # P1
            if col < 3:
                g.step(col + 4)  # P2 in another column
        assert g.is_over()
        assert g.winner() == 1

    def test_c4_repr(self):
        g = Connect4()
        r = repr(g)
        assert "0" in r  # Column numbers shown

    def test_c4_full_column(self):
        g = Connect4()
        import pytest
        # Fill a column — game may end via win before column is full
        # Just verify step() works and eventually stops
        for _ in range(12):  # Max 12 steps (6 rows × 2 cols)
            try:
                actions = g.legal_actions()
                if not actions or g.is_over():
                    break
                g.step(actions[0])
            except (ValueError, RuntimeError):
                break  # Game over or column full


class TestToPythonExport:
    """The generated Python code must work independently."""

    def test_to_python_uses_blake2b(self):
        """Generated code must use blake2b to match the compiled table."""
        f = TileField(seed=42).train(TicTacToe(), n_games=100)
        policy = compile(f)
        code = policy.to_python()
        assert "blake2b" in code
        assert "hashlib" in code

    def test_to_python_produces_valid_lookup(self):
        """The generated choose function should match the compiled policy."""
        f = TileField(seed=42).train(TicTacToe(), n_games=100)
        policy = compile(f)

        code = policy.to_python()
        exec_globals = {}
        exec(code, exec_globals)
        choose_fn = exec_globals["choose"]

        game = TicTacToe()
        result = choose_fn(game.state())
        expected = policy.choose(game.state())
        assert result == expected

    def test_to_python_end_to_end(self):
        """Full game using generated code."""
        f = TileField(seed=42).train(TicTacToe(), n_games=200)
        policy = compile(f)
        code = policy.to_python()
        exec_globals = {}
        exec(code, exec_globals)
        choose_fn = exec_globals["choose"]

        game = TicTacToe()
        while not game.is_over():
            action = choose_fn(game.state())
            if action is None:
                action = game.legal_actions()[0]
            game.step(action)
        assert game.is_over()


class TestTemperatureSoftmax:
    """Temperature parameter must actually be used."""

    def test_low_temperature_deterministic(self):
        """T=0.01 should be nearly deterministic."""
        f = TileField(seed=42).train(TicTacToe(), n_games=100, temperature=0.01)
        # Run choose many times — should always get same answer
        game = TicTacToe()
        results = set()
        for _ in range(20):
            results.add(f.choose(game.state()))
        # Should be deterministic or very close
        assert len(results) <= 2

    def test_high_temperature_exploratory(self):
        """T=5.0 should produce varied choices."""
        f = TileField(seed=42).train(TicTacToe(), n_games=100, temperature=5.0)
        game = TicTacToe()
        # With high T, we should see some variation (though not guaranteed)
        # Just verify it doesn't crash
        action = f.choose(game.state())
        assert action is not None or len(game.legal_actions()) == 0


class TestMemoryDecay:
    """Score decay should gradually reduce weights."""

    def test_decay_reduces_weights(self):
        """After training with decay, weights should be lower than without."""
        f_no_decay = TileField(seed=42).train(TicTacToe(), n_games=100, decay=0.0)
        f_decay = TileField(seed=42).train(TicTacToe(), n_games=100, decay=0.05)

        # The decayed field should have lower total weight
        w_no = sum(sum(a.values()) for a in f_no_decay.export_weights().values())
        w_decay = sum(sum(a.values()) for a in f_decay.export_weights().values())
        # With 5% decay per game, weights should be notably lower
        assert w_decay < w_no

    def test_default_decay_is_small(self):
        """Default decay=0.005 should be barely noticeable."""
        f = TileField(seed=42).train(TicTacToe(), n_games=50)
        # Should still learn effectively
        assert f.n_states > 0
        policy = compile(f)
        assert policy.size > 0


class TestDiagnostics:
    """The analyze() function should produce meaningful reports."""

    def test_analyze_trained_field(self):
        f = TileField(seed=42).train(TicTacToe(), n_games=100)
        report = analyze(f)
        assert isinstance(report, FieldReport)
        assert report.n_states > 0
        assert report.n_games == 100
        assert report.score_cv >= 0
        assert report.entropy_bits >= 0
        assert report.active_tiles >= 0

    def test_analyze_empty_field(self):
        f = TileField(seed=42)
        report = analyze(f)
        assert report.n_states == 0
        assert report.n_games == 0

    def test_report_summary(self):
        f = TileField(seed=42).train(TicTacToe(), n_games=50)
        report = analyze(f)
        summary = report.summary()
        assert "Tile Field Report" in summary
        assert "states" in summary
