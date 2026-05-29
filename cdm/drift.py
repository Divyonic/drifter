"""Drift scoring for the Context Drift Monitor.

The :class:`DriftEngine` turns embedded turns into :class:`~cdm.models.DriftScore`
measurements. Drift is measured as cosine distance from two references:

* the fixed *anchor* goal (the conversation's north star), and
* a rolling *reference* (typically a centroid of recent goal-state text), which
  may be ``None`` early in a session — in that case we fall back to the anchor.

A turn is flagged as drifting when *either* distance exceeds the threshold.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from cdm.config import DEFAULT_THRESHOLD, ROLLING_WINDOW
from cdm.embeddings import cosine_distance
from cdm.models import DriftScore

__all__ = [
    "DriftEngine",
    "baseline_stats",
    "cusum_changepoint",
    "forecast_cross",
    "biggest_jump",
]


class DriftEngine:
    """Compute per-turn drift scores and reference centroids."""

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        window: int = ROLLING_WINDOW,
    ) -> None:
        """Create an engine.

        Args:
            threshold: Cosine-distance value above which a turn is flagged.
            window: Number of most-recent embeddings used by
                :meth:`rolling_reference`.
        """
        self.threshold = float(threshold)
        self.window = int(window)

    def score_turn(
        self,
        message_emb: List[float],
        anchor_emb: List[float],
        reference_emb: Optional[List[float]],
        turn_id: int,
        session_id: str,
    ) -> DriftScore:
        """Score a single turn against the anchor and rolling reference.

        ``drift_from_anchor`` is the cosine distance between the message and the
        anchor embedding. ``drift_from_reference`` uses ``reference_emb`` when it
        is provided, otherwise it falls back to the anchor distance. The turn is
        flagged (``is_drift_high``) when *either* distance exceeds the threshold.

        Args:
            message_emb: Unit-normalised embedding of the turn's text.
            anchor_emb: Unit-normalised embedding of the anchor goal.
            reference_emb: Unit-normalised rolling reference embedding, or
                ``None`` to reuse the anchor distance.
            turn_id: 0-based turn index within the session.
            session_id: Owning session identifier.

        Returns:
            A populated :class:`~cdm.models.DriftScore`.
        """
        drift_from_anchor = cosine_distance(message_emb, anchor_emb)
        if reference_emb:
            drift_from_reference = cosine_distance(message_emb, reference_emb)
        else:
            drift_from_reference = drift_from_anchor
        is_drift_high = (
            drift_from_anchor > self.threshold
            or drift_from_reference > self.threshold
        )
        return DriftScore(
            turn_id=turn_id,
            session_id=session_id,
            drift_from_reference=drift_from_reference,
            drift_from_anchor=drift_from_anchor,
            is_drift_high=is_drift_high,
        )

    @staticmethod
    def centroid(embeddings: List[List[float]]) -> List[float]:
        """Return the L2-normalised mean of ``embeddings``.

        Args:
            embeddings: A list of equal-length vectors.

        Returns:
            The unit-normalised mean vector, or ``[]`` for empty input (or when
            the mean is the zero vector).
        """
        if not embeddings:
            return []
        matrix = np.asarray(embeddings, dtype=float)
        mean = matrix.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm == 0.0 or not math.isfinite(norm):
            return []
        return (mean / norm).tolist()

    def rolling_reference(self, embeddings: List[List[float]]) -> List[float]:
        """Centroid of the last ``self.window`` embeddings (unit-normalised).

        Args:
            embeddings: Chronological list of turn embeddings.

        Returns:
            The unit-normalised centroid of the trailing window, or ``[]`` when
            there are no embeddings.
        """
        if not embeddings:
            return []
        window = max(1, self.window)
        recent = embeddings[-window:]
        return self.centroid(recent)

    @staticmethod
    def smooth(values: List[float], window: int) -> List[float]:
        """Apply a trailing moving average to ``values``.

        Each output element is the mean of up to ``window`` preceding values
        (inclusive), so the series length is preserved and early values use a
        shorter window. ``window <= 1`` returns an unchanged copy.

        Args:
            values: The series to smooth.
            window: Trailing window size.

        Returns:
            A new list of the same length as ``values``.
        """
        if window <= 1 or len(values) == 0:
            return list(values)
        out: List[float] = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            chunk = values[start : i + 1]
            out.append(sum(chunk) / len(chunk))
        return out


# --------------------------------------------------------------------------- #
# Self-calibrating analytics (pure functions; operate on a drift-vs-anchor series)
# --------------------------------------------------------------------------- #
def baseline_stats(series: List[float], k: int = 3) -> tuple:
    """Return ``(mean, std)`` of the conversation's early on-goal drift.

    The opening turn often equals the anchor (distance ~0), so a single leading
    near-zero is skipped. ``std`` is floored to a small epsilon so downstream
    z-scores stay finite. This is the *self-calibrating* baseline: "normal" is
    learned per-conversation rather than assumed, which makes detection robust to
    whichever embedder/scale is in use.
    """
    vals = list(series)
    if vals and vals[0] <= 1e-6:
        vals = vals[1:]
    early = vals[:k] if k > 0 else vals
    if not early:
        return 0.0, 0.05
    arr = np.asarray(early, dtype=float)
    return float(arr.mean()), max(float(arr.std()), 0.03)


def cusum_changepoint(
    series: List[float], mean: float, std: float, slack: float = 0.4, h: float = 3.0
) -> Optional[int]:
    """One-sided CUSUM detector for a sustained upward shift.

    Accumulates standardised excursions above ``mean`` (minus ``slack``) and
    alarms when the running sum exceeds ``h``. Returns the series index where the
    shift first alarms, or ``None`` if the conversation never sustainedly departs
    from its baseline. More principled than a single fixed cutoff: it ignores
    one-off spikes and fires on a real regime change.
    """
    if not series:
        return None
    denom = std or 1.0
    total = 0.0
    for i, x in enumerate(series):
        total = max(0.0, total + (x - mean) / denom - slack)
        if total > h:
            return i
    return None


def forecast_cross(
    series: List[float], threshold: float, window: int = 5
) -> Optional[float]:
    """Estimate how many turns until ``series`` crosses ``threshold``.

    Fits a least-squares slope over the last ``window`` points and extrapolates.
    Returns ``0.0`` if already at/above the threshold, a positive number of turns
    ahead if it is rising toward it, or ``None`` if flat/declining or too short.
    """
    if len(series) < 2:
        return None
    last = float(series[-1])
    if last >= threshold:
        return 0.0
    recent = series[-window:] if window > 0 else series
    if len(recent) < 2:
        return None
    xs = np.arange(len(recent), dtype=float)
    slope = float(np.polyfit(xs, np.asarray(recent, dtype=float), 1)[0])
    if slope <= 1e-4:
        return None
    return (threshold - last) / slope


def biggest_jump(series: List[float]) -> Optional[int]:
    """Index of the largest single-step increase (the turn that drove drift most).

    Returns ``None`` when the series never rises step-to-step.
    """
    if len(series) < 2:
        return None
    deltas = [series[i] - series[i - 1] for i in range(1, len(series))]
    j = int(np.argmax(deltas))
    return j + 1 if deltas[j] > 0 else None
