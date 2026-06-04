"""Training configuration — immutable, prevents parameter drift.

This module exists because the builder found that training parameters
scattered across __init__, train(), and internal fields caused 3 bugs:
1. weight_decay and score_decay were conflated (one field, two concepts)
2. temperature was stored but never wired
3. explore_rate had no consistent default

TrainingConfig makes every training parameter explicit and immutable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training configuration for TileField.

    All parameters have validated-optimal defaults from experiments:
    - T=0.3: best for greedy exploitation (temperature sweep experiment)
    - score_decay=0.005: free insurance, recovers in 30 games (decay experiment)
    - weight_decay=0.95: per-update retention, prevents runaway weights
    - explore_rate=0.3: balances discovery vs exploitation

    Example::

        config = TrainingConfig(temperature=0.5, score_decay=0.01)
        field = TileField(config=config)
        field.train(game, n_games=500)

        # Override at train time
        field.train(game, n_games=200, config=config.with_overrides(temperature=0.1))
    """

    # Learning parameters
    learning_rate: float = 0.1
    weight_decay: float = 0.95        # Per-update: keep 95% of old weight
    score_decay: float = 0.005         # Per-game: lose 0.5% (free insurance)
    temperature: float = 0.3           # Softmax T (validated optimal)
    explore_rate: float = 0.3          # Epsilon-greedy exploration

    # Evolution parameters
    n_generations: int = 10
    population: int = 20

    def with_overrides(self, **kwargs: float) -> "TrainingConfig":
        """Return a new config with specified overrides.

        Example::

            aggressive = config.with_overrides(temperature=0.1, explore_rate=0.1)
        """
        d = asdict(self)
        d.update(kwargs)
        return TrainingConfig(**d)

    def __repr__(self) -> str:
        return (
            f"TrainingConfig("
            f"T={self.temperature}, "
            f"lr={self.learning_rate}, "
            f"decay={self.score_decay})"
        )


# Singleton for default configuration
DEFAULT_CONFIG = TrainingConfig()
