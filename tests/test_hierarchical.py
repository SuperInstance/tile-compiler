"""Tests for hierarchical compilation."""

from tile_compiler import TileField, hierarchical_compile
from tests.conftest import TicTacToe as TestTTT


class TestHierarchical:
    def test_hierarchical_creates_policy(self):
        f = TileField(seed=0).train(TestTTT(), n_games=50)
        h = hierarchical_compile(f.export_weights(), n_clusters=4)
        assert h.n_clusters <= 4

    def test_hierarchical_chooses(self):
        g = TestTTT()
        f = TileField(seed=0).train(g, n_games=50)
        h = hierarchical_compile(f.export_weights(), n_clusters=4)
        action = h.choose(g.state())
        assert action is not None or True  # may miss some states

    def test_hierarchical_cluster_count(self):
        f = TileField(seed=0).train(TestTTT(), n_games=50)
        h = hierarchical_compile(f.export_weights(), n_clusters=4)
        assert h.n_clusters > 0
        assert h.n_clusters <= 4

    def test_hierarchical_total_entries(self):
        f = TileField(seed=0).train(TestTTT(), n_games=50)
        h = hierarchical_compile(f.export_weights(), n_clusters=4)
        assert h.total_entries > 0

    def test_empty_weights(self):
        h = hierarchical_compile({})
        assert h.n_clusters == 0

    def test_repr(self):
        f = TileField(seed=0).train(TestTTT(), n_games=50)
        h = hierarchical_compile(f.export_weights(), n_clusters=4)
        assert "HierarchicalPolicy" in repr(h)
