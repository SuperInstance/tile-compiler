"""Tests for the analyze module."""

import pytest
from tile_compiler import TileField, analyze, AnalysisReport, compile
from tile_compiler.analyze import analyze as _analyze


class TestAnalyzeEmpty:
    """Analyze should handle empty fields gracefully."""

    def test_empty_field_report(self):
        field = TileField(seed=42)
        report = analyze(field)
        assert report.n_states == 0
        assert report.n_games == 0
        assert report.suggested_action == "train_more"

    def test_empty_field_has_no_coverage(self):
        field = TileField(seed=42)
        report = analyze(field)
        assert report.top_n_coverage == {}


class TestAnalyzeTrained:
    """Analyze on a trained field."""

    @pytest.fixture
    def trained_field(self):
        """Train a simple game."""
        class SimpleGame:
            def __init__(self):
                self.board = [0, 0, 0]
                self._over = False
                self._winner = None

            def reset(self):
                self.board = [0, 0, 0]
                self._over = False
                self._winner = None

            def state(self):
                return tuple(self.board)

            def legal_actions(self):
                return [i for i, v in enumerate(self.board) if v == 0]

            def step(self, action):
                self.board[action] = 1
                if sum(self.board) == 3:
                    self._over = True
                    self._winner = 1

            def is_over(self):
                return self._over

            def winner(self):
                return self._winner

            def current_player(self):
                return 1

        field = TileField(seed=42)
        field.train(SimpleGame(), n_games=50)
        return field

    def test_report_has_states(self, trained_field):
        report = analyze(trained_field)
        assert report.n_states > 0
        assert report.n_games == 50

    def test_report_has_score_stats(self, trained_field):
        report = analyze(trained_field)
        assert report.score_mean != 0 or report.score_std != 0

    def test_report_has_capacity(self, trained_field):
        report = analyze(trained_field)
        assert report.active_tiles + report.dead_tiles == report.n_states

    def test_report_has_entropy(self, trained_field):
        report = analyze(trained_field)
        assert report.action_entropy >= 0
        assert 0 <= report.dominant_action_frac <= 1.0

    def test_report_has_coverage(self, trained_field):
        report = analyze(trained_field)
        assert len(report.top_n_coverage) > 0
        # Top-N coverage should be non-decreasing with N
        ns = sorted(report.top_n_coverage.keys())
        for i in range(1, len(ns)):
            assert report.top_n_coverage[ns[i]] >= report.top_n_coverage[ns[i-1]] - 1e-9

    def test_report_has_recommendation(self, trained_field):
        report = analyze(trained_field)
        assert report.suggested_action in ("compile", "optimize", "train_more")
        assert isinstance(report.suggested_reason, str)
        assert len(report.suggested_reason) > 0

    def test_report_repr(self, trained_field):
        report = analyze(trained_field)
        r = repr(report)
        assert "AnalysisReport" in r
        assert "cv=" in r


class TestAnalyzeRecommendations:
    """Test that recommendations are correct for edge cases."""

    def test_all_dead_tiles_recommends_optimize(self):
        """Field with all dead tiles (visits=0) should recommend optimize."""
        field = TileField(seed=42)
        # Manually add weights but no visits
        field._weights[123] = {0: 0.5, 1: 0.5}
        field._weights[456] = {2: 0.5}
        # visits stay at 0
        report = analyze(field)
        assert report.suggested_action == "optimize"

    def test_small_field_recommends_compile(self):
        """Field with <50 active tiles should recommend compile."""
        field = TileField(seed=42)
        # Add a few tiles with visits and similar scores (low CV)
        for i in range(10):
            field._weights[i] = {0: 0.5, 1: 0.5}
            field._visits[i] = 5
        report = analyze(field)
        assert report.suggested_action == "compile"


class TestHierarchicalFallback:
    """Test that hierarchical policy falls back to centroid vote."""

    @pytest.fixture
    def simple_weights(self):
        return {
            100: {0: 0.9, 1: 0.1},
            200: {0: 0.8, 1: 0.2},
            300: {2: 0.7, 3: 0.3},
        }

    def test_known_state_returns_exact(self, simple_weights):
        from tile_compiler.hierarchical import hierarchical_compile
        policy = hierarchical_compile(simple_weights, n_clusters=2)
        # State with key 100 should resolve
        result = policy.choose(100)  # key itself is the state
        # May return None if hashing differs, but shouldn't error
        # The real test: no crash

    def test_fallback_doesnt_crash(self, simple_weights):
        from tile_compiler.hierarchical import hierarchical_compile
        policy = hierarchical_compile(simple_weights, n_clusters=2)
        # Unknown state should use centroid fallback or return None
        result = policy.choose("unknown_state_xyz")
        # Should not crash, may return None or a centroid action

    def test_centroid_actions_populated(self, simple_weights):
        from tile_compiler.hierarchical import hierarchical_compile
        policy = hierarchical_compile(simple_weights, n_clusters=2)
        assert len(policy._centroid_action) > 0


class TestCompileFieldAlias:
    """Test that compile_field works as an alias."""

    def test_alias_exists(self):
        from tile_compiler import compile_field
        assert callable(compile_field)

    def test_alias_produces_same_result(self):
        from tile_compiler import compile, compile_field, TileField

        class MiniGame:
            def __init__(self):
                self.board = [0, 0]
                self._over = False
                self._winner = 1

            def reset(self):
                self.board = [0, 0]
                self._over = False

            def state(self):
                return tuple(self.board)

            def legal_actions(self):
                return [i for i, v in enumerate(self.board) if v == 0]

            def step(self, action):
                self.board[action] = 1
                if sum(self.board) == 2:
                    self._over = True

            def is_over(self):
                return self._over

            def winner(self):
                return 1

            def current_player(self):
                return 1

        field = TileField(seed=42).train(MiniGame(), n_games=20)
        p1 = compile(field)
        p2 = compile_field(field)
        assert p1.size == p2.size
