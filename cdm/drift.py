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

__all__ = ["DriftEngine"]


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
