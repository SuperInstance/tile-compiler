"""TileField: train on games, evolve weights, pick moves."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, Optional

# Lazy import to avoid circular dependency


class TileField:
    """A trainable field that maps board states to action weights.

    Train it on any game by playing matches, then compile it into a
    fast lookup-table policy.

    Example::

        field = TileField()
        field.train(my_game, n_games=1000)
        policy = field.compile()
        action = policy.choose(state)
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        # board_hash -> {action: weight}
        self._weights: dict[int, dict[Any, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._visits: dict[int, int] = defaultdict(int)
        self._games_played: int = 0
        self._learning_rate: float = 0.1
        self._decay: float = 0.95

    @property
    def n_states(self) -> int:
        """Number of unique states with weights."""
        return len(self._weights)

    @property
    def games_played(self) -> int:
        return self._games_played

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        game: Any,
        n_games: int = 500,
        *,
        explore_rate: float = 0.3,
        learning_rate: Optional[float] = None,
        decay: Optional[float] = None,
    ) -> "TileField":
        """Train by playing ``n_games`` self-play matches.

        Parameters
        ----------
        game:
            Any object implementing the Game protocol (``reset()``,
            ``legal_actions()``, ``step(action)``, ``is_over()``,
            ``winner()``, ``current_player()``, ``state()``).
        n_games:
            Number of self-play games.
        explore_rate:
            Probability of choosing a random legal action (epsilon-greedy).
        learning_rate:
            Step size for weight updates (default: 0.1).
        decay:
            Weight decay per update (default: 0.95).

        Returns
        -------
        self (for chaining)
        """
        if learning_rate is not None:
            self._learning_rate = learning_rate
        if decay is not None:
            self._decay = decay

        for _ in range(n_games):
            self._play_one(game, explore_rate)
            self._games_played += 1
        return self

    def evolve(self, generations: int = 10, population: int = 20) -> "TileField":
        """Evolve weights via a simple mutation-selection loop.

        This is a lightweight evolutionary optimisation that perturbs
        the current weight table and keeps improvements.
        """
        for _ in range(generations):
            candidates = [self._mutate() for _ in range(population)]
            # evaluate by self-play score
            best = max(candidates, key=lambda c: c._total_weight())
            if best._total_weight() >= self._total_weight():
                self._weights = best._weights
                self._visits = best._visits
        return self

    def choose(self, state: Any) -> Any:
        """Pick the best action for *state* using current weights."""
        key = self._hash_state(state)
        action_weights = self._weights.get(key)
        if not action_weights:
            return None
        return max(action_weights, key=lambda a: action_weights[a])

    # ------------------------------------------------------------------
    # Compilation shortcuts
    # ------------------------------------------------------------------

    def compile(self) -> "CompiledPolicy":
        """Compile into a :class:`CompiledPolicy` lookup table."""
        from tile_compiler.compiler import compile as compile_field
        return compile_field(self)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _play_one(self, game: Any, explore_rate: float) -> None:
        """Play a single self-play game, updating weights."""
        game.reset()
        trajectory: list[tuple[int, Any, Any]] = []  # (state_key, action, player)

        while not game.is_over():
            state = game.state()
            key = self._hash_state(state)
            actions = game.legal_actions()
            if not actions:
                break

            if self._rng.random() < explore_rate:
                action = self._rng.choice(actions)
            else:
                action = self._choose_weighted(key, actions)

            player = game.current_player()
            trajectory.append((key, action, player))
            game.step(action)

        # Update weights based on outcome
        winner = game.winner() if hasattr(game, "winner") else None
        for state_key, action, player in trajectory:
            self._visits[state_key] += 1
            reward = self._reward(player, winner)
            self._update_weight(state_key, action, reward)

    def _choose_weighted(self, key: int, actions: list[Any]) -> Any:
        """Choose from *actions* weighted by current field values."""
        weights = self._weights.get(key, {})
        if not weights:
            return self._rng.choice(actions)
        best = max(actions, key=lambda a: weights.get(a, 0.0))
        return best

    def _update_weight(self, key: int, action: Any, reward: float) -> None:
        """Apply a reward-based weight update."""
        old = self._weights[key].get(action, 0.0)
        self._weights[key][action] = old * self._decay + self._learning_rate * reward

    def _reward(self, player: Any, winner: Any) -> float:
        """Compute reward for a player given the game winner."""
        if winner is None:
            return 0.0
        return 1.0 if player == winner else -1.0

    def _mutate(self) -> "TileField":
        """Create a perturbed copy of this field."""
        child = TileField.__new__(TileField)
        child._rng = random.Random()
        child._weights = defaultdict(lambda: defaultdict(float))
        child._visits = dict(self._visits)
        child._games_played = self._games_played
        child._learning_rate = self._learning_rate
        child._decay = self._decay

        for key, actions in self._weights.items():
            for action, weight in actions.items():
                child._weights[key][action] = weight + self._rng.gauss(0, 0.05)
        return child

    def _total_weight(self) -> float:
        return sum(
            sum(actions.values()) for actions in self._weights.values()
        )

    @staticmethod
    def _hash_state(state: Any) -> int:
        """Hash a game state. Handles tuples, lists, and strings."""
        if isinstance(state, (tuple, list)):
            return hash(tuple(state))
        return hash(state)

    def export_weights(self) -> dict[int, dict[Any, float]]:
        """Return a copy of the weight table."""
        return {
            k: dict(v) for k, v in self._weights.items()
        }
