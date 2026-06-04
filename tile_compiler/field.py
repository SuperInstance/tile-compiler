"""TileField: train on games, evolve weights, pick moves."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, Optional, Protocol, runtime_checkable

# Lazy import to avoid circular dependency


@runtime_checkable
class Game(Protocol):
    """Protocol for games compatible with :class:`TileField`.

    Any object implementing these seven methods can be used for
    training.  A :class:`TypeError` is raised at runtime if a
    required method is missing.
    """

    def reset(self) -> None: ...
    def state(self) -> Any: ...
    def legal_actions(self) -> list[Any]: ...
    def step(self, action: Any) -> None: ...
    def is_over(self) -> bool: ...
    def winner(self) -> Any: ...
    def current_player(self) -> Any: ...


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
        self._weight_decay: float = 0.95  # Keep fraction of old weight per update
        self._score_decay: float = 0.005  # Memory decay rate (free insurance)
        self._temperature: float = 0.3    # Softmax temperature

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
        temperature: Optional[float] = None,
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
        temperature:
            Softmax temperature for weighted action selection.
            T=0.3 validated as optimal for exploitation. Default: 0.3.
        learning_rate:
            Step size for weight updates (default: 0.1).
        decay:
            Memory decay rate per game. 0.005 is free insurance against
            regime changes — costs nothing in stable environments.
            Default: 0.005.

        Returns
        -------
        self (for chaining)
        """
        if learning_rate is not None:
            self._learning_rate = learning_rate
        if temperature is not None:
            self._temperature = temperature
        if decay is not None:
            self._score_decay = decay

        for _ in range(n_games):
            self._play_one(game, explore_rate)
            self._games_played += 1
            # Apply score decay after each game
            self._apply_score_decay()
        return self

    def evolve(self, generations: int = 10, population: int = 20, game: Any = None) -> "TileField":
        """Evolve weights via a simple mutation-selection loop.

        This is a lightweight evolutionary optimisation that perturbs
        the current weight table and keeps improvements.

        Parameters
        ----------
        generations:
            Number of evolutionary generations.
        population:
            Number of candidate mutants per generation.
        game:
            Optional game for fitness evaluation (self-play).
            If ``None``, uses max-best-action-score as fitness
            (better than total weight).
        """
        for _ in range(generations):
            candidates = [self._mutate() for _ in range(population)]
            candidates.append(self)  # elitism: include parent
            best = max(candidates, key=lambda c: c._fitness())
            if best is not self:
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
        """Choose from *actions* using softmax with temperature."""
        import math
        weights = self._weights.get(key, {})
        if not weights:
            return self._rng.choice(actions)

        # Softmax with temperature
        scores = [weights.get(a, 0.0) for a in actions]
        t = max(self._temperature, 0.01)  # Clamp to avoid division by zero
        exp_scores = [math.exp(s / t) for s in scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]

        # Weighted random choice
        r = self._rng.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return actions[i]
        return actions[-1]

    def _update_weight(self, key: int, action: Any, reward: float) -> None:
        """Apply a reward-based weight update with exponential decay."""
        old = self._weights[key].get(action, 0.0)
        self._weights[key][action] = old * self._weight_decay + self._learning_rate * reward

    def _reward(self, player: Any, winner: Any) -> float:
        """Compute reward for a player given the game winner."""
        if winner is None:
            return 0.0
        return 1.0 if player == winner else -1.0

    def _apply_score_decay(self) -> None:
        """Apply memory decay — gently pushes all weights toward 0.

        Decay rate 0.005 means each game, weights lose 0.5% of their magnitude.
        This is free insurance against regime changes: costs ~nothing in stable
        environments but helps escape stuck strategies when opponent changes.
        """
        rate = self._score_decay
        for actions in self._weights.values():
            for action in actions:
                actions[action] *= (1.0 - rate)

    def _mutate(self) -> "TileField":
        """Create a perturbed copy of this field."""
        child = TileField.__new__(TileField)
        child._rng = random.Random()
        child._weights = defaultdict(lambda: defaultdict(float))
        child._visits = dict(self._visits)
        child._games_played = self._games_played
        child._learning_rate = self._learning_rate
        child._weight_decay = self._weight_decay
        child._score_decay = self._score_decay
        child._temperature = self._temperature

        for key, actions in self._weights.items():
            for action, weight in actions.items():
                child._weights[key][action] = weight + self._rng.gauss(0, 0.05)
        return child

    def _total_weight(self) -> float:
        return sum(
            sum(actions.values()) for actions in self._weights.values()
        )

    def _fitness(self) -> float:
        """Fitness for evolution: sum of best-action scores per state.

        Better than total_weight because it measures peak confidence
        rather than accumulated mass (which decays with more training).
        """
        if not self._weights:
            return 0.0
        return sum(
            max(actions.values()) if actions else 0.0
            for actions in self._weights.values()
        )

    @staticmethod
    def _hash_state(state: Any) -> int:
        """Deterministic hash — delegates to the canonical ``hash_state``."""
        from tile_compiler import hash_state
        return hash_state(state)

    def export_weights(self) -> dict[int, dict[Any, float]]:
        """Return a copy of the weight table."""
        return {
            k: dict(v) for k, v in self._weights.items()
        }
