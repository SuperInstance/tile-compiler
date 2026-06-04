"""Tests for serialization and protocol."""

import json
import tempfile
from pathlib import Path

from tile_compiler import TileField, compile, CompiledPolicy, validate_game
from tile_compiler.protocol import Game


class TestCompiledPolicySerialization:
    def test_save_and_load(self, tmp_path):
        f = TileField(seed=42)
        from tests.conftest import TicTacToe
        f.train(TicTacToe(), n_games=100)
        policy = compile(f)

        path = str(tmp_path / "policy.json")
        policy.save(path)

        loaded = CompiledPolicy.load(path)
        assert loaded.size == policy.size

        game = TicTacToe()
        assert loaded.choose(game.state()) == policy.choose(game.state())

    def test_save_creates_valid_json(self, tmp_path):
        f = TileField(seed=42)
        from tests.conftest import TicTacToe
        f.train(TicTacToe(), n_games=50)
        policy = compile(f)

        path = str(tmp_path / "policy.json")
        policy.save(path)

        with open(path) as fh:
            data = json.load(fh)
        assert data["version"] == 1
        assert data["size"] == policy.size
        assert isinstance(data["table"], dict)

    def test_to_python_generates_code(self):
        f = TileField(seed=42)
        from tests.conftest import TicTacToe
        f.train(TicTacToe(), n_games=50)
        policy = compile(f)

        code = policy.to_python()
        assert "def choose" in code
        assert "_table" in code
        assert "import" not in code or "import hashlib" in code  # Only stdlib allowed


class TestProtocol:
    def test_validate_good_game(self):
        from tests.conftest import TicTacToe
        issues = validate_game(TicTacToe())
        assert issues == []

    def test_validate_bad_game(self):
        class BadGame:
            def reset(self): pass
            # Missing: state, legal_actions, step, is_over, winner

        issues = validate_game(BadGame())
        assert len(issues) > 0
        assert any("state" in i for i in issues)

    def test_game_protocol_is_runtime_checkable(self):
        from tests.conftest import TicTacToe
        assert isinstance(TicTacToe(), Game)
