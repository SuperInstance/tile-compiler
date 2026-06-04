"""Diagnostics and analysis for trained TileFields.

Exposes experimental insights (conservation, holographic bound, capacity)
as a user-facing diagnostic tool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from tile_compiler.field import TileField


@dataclass
class AnalysisReport:
    """Diagnostic report for a trained TileField."""

    # Core stats
    n_states: int = 0
    n_games: int = 0

    # Conservation (CV < 0.02 = healthy)
    score_cv: float = 0.0
    score_mean: float = 0.0
    score_std: float = 0.0

    # Capacity
    active_tiles: int = 0
    dead_tiles: int = 0
    dead_fraction: float = 0.0

    # Interference
    action_entropy: float = 0.0
    dominant_action_frac: float = 0.0

    # Holographic bound: {n: fraction of total best-action score}
    top_n_coverage: dict[int, float] = field(default_factory=dict)

    # Top actions distribution
    top_actions: dict[Any, int] = field(default_factory=dict)

    # Decision entropy per state
    entropy_bits: float = 0.0

    # Recommendations
    suggested_action: str = "train_more"
    suggested_reason: str = ""

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Tile Field Report ({self.n_states} states, {self.n_games} games)",
            f"  Score: μ={self.score_mean:.3f} σ={self.score_std:.3f} CV={self.score_cv:.4f}",
            f"  Active tiles: {self.active_tiles} ({self.dead_fraction:.0%} dead)",
            f"  Entropy: {self.entropy_bits:.3f} bits",
            f"  Holographic: {self.top_n_coverage.get(5, 0):.0%} in top 5 tiles",
            f"  Recommendation: {self.suggested_action} — {self.suggested_reason}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"AnalysisReport("
            f"states={self.n_states}, "
            f"cv={self.score_cv:.4f}, "
            f"dead={self.dead_fraction:.0%}, "
            f"recommend={self.suggested_action})"
        )


def analyze(
    field: "TileField",
    top_n_values: list[int] | None = None,
) -> AnalysisReport:
    """Analyze a trained TileField and produce a diagnostic report.

    Parameters
    ----------
    field:
        A trained TileField.
    top_n_values:
        List of n-values for holographic coverage analysis.
        Default: [1, 5, 10, 50, 100].

    Returns
    -------
    AnalysisReport

    Example
    -------
    >>> from tile_compiler import TileField, analyze
    >>> field = TileField().train(game, n_games=500)
    >>> report = analyze(field)
    >>> print(f"Conservation CV: {report.score_cv:.4f}")
    >>> print(f"Recommendation: {report.suggested_action}")
    """
    if top_n_values is None:
        top_n_values = [1, 5, 10, 50, 100]

    weights = field.export_weights()
    visits = dict(field._visits)
    n_games = field.games_played

    report = AnalysisReport(n_games=n_games)

    if not weights:
        report.suggested_action = "train_more"
        report.suggested_reason = "Field has no trained states"
        return report

    report.n_states = len(weights)

    # --- Score statistics ---
    # Use per-state best-action scores for CV (not raw all scores).
    # Raw scores include both positive (wins) and negative (losses),
    # so the mean ≈ 0 and CV → ∞. Best-action scores measure
    # decision confidence — the conservation law applies to these.
    best_action_scores: list[float] = []
    top_actions: dict[Any, int] = {}
    entropies: list[float] = []

    for actions in weights.values():
        if not actions:
            continue

        # Best action per state
        best = max(actions, key=lambda a: actions[a])
        best_score = actions[best]
        best_action_scores.append(best_score)
        top_actions[best] = top_actions.get(best, 0) + 1

        # Per-state entropy
        if len(actions) >= 2:
            scores = list(actions.values())
            total = sum(abs(s) for s in scores)
            if total > 1e-12:
                probs = [abs(s) / total for s in scores]
                h = -sum(p * math.log2(p) for p in probs if p > 1e-12)
                entropies.append(h)

    if best_action_scores:
        report.score_mean = sum(best_action_scores) / len(best_action_scores)
        variance = sum((s - report.score_mean) ** 2 for s in best_action_scores) / len(best_action_scores)
        report.score_std = math.sqrt(variance)
        report.score_cv = report.score_std / abs(report.score_mean) if abs(report.score_mean) > 1e-12 else 0.0

    report.top_actions = dict(sorted(top_actions.items(), key=lambda x: -x[1]))
    report.entropy_bits = sum(entropies) / len(entropies) if entropies else 0.0

    # --- Capacity ---
    active = sum(1 for k in weights if visits.get(k, 0) > 0)
    dead = sum(1 for k in weights if visits.get(k, 0) == 0)
    report.active_tiles = active
    report.dead_tiles = dead
    total = active + dead
    report.dead_fraction = dead / total if total > 0 else 0.0

    # --- Interference: action entropy ---
    total_tiles = sum(top_actions.values()) if top_actions else 1
    if top_actions:
        entropy = 0.0
        for count in top_actions.values():
            p = count / total_tiles
            if p > 0:
                entropy -= p * math.log2(p)
        report.action_entropy = entropy
        report.dominant_action_frac = max(top_actions.values()) / total_tiles

    # --- Holographic bound ---
    tile_scores: list[tuple[int, float]] = []
    for key, actions in weights.items():
        if actions:
            best_score = max(actions.values())
            tile_scores.append((key, best_score))

    tile_scores.sort(key=lambda x: x[1], reverse=True)
    total_score = sum(s for _, s in tile_scores)

    if total_score > 0:
        for n in top_n_values:
            if n >= len(tile_scores):
                report.top_n_coverage[n] = 1.0
            else:
                top_n_sum = sum(s for _, s in tile_scores[:n])
                report.top_n_coverage[n] = top_n_sum / total_score

    # --- Recommendation ---
    if not weights:
        report.suggested_action = "train_more"
        report.suggested_reason = "Field has no trained states"
    elif report.dead_fraction > 0.5:
        report.suggested_action = "optimize"
        report.suggested_reason = (
            f"{report.dead_fraction:.0%} of tiles are dead — "
            "dead code elimination will help massively"
        )
    elif report.active_tiles < 50 and report.score_cv < 0.5:
        report.suggested_action = "compile"
        report.suggested_reason = (
            f"Only {report.active_tiles} active tiles — "
            "simple compilation is sufficient"
        )
    elif report.score_cv > 0.5:
        report.suggested_action = "train_more"
        report.suggested_reason = (
            f"Score CV={report.score_cv:.4f} (>0.5) — "
            "field hasn't converged yet"
        )
    else:
        report.suggested_action = "optimize"
        report.suggested_reason = (
            f"{report.active_tiles} active tiles with "
            f"CV={report.score_cv:.4f} — "
            "5-pass pipeline recommended"
        )

    return report


# Backward compat alias
FieldReport = AnalysisReport
