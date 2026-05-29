"""Tests for :mod:`cdm.drift` — offline, hashing-embedder compatible.

These tests rely only on the mathematical definition of cosine distance
(``1 - cosine_similarity``, clamped to ``[0, 2]``) using simple unit vectors:
identical vectors -> distance 0, orthogonal -> 1, opposite -> 2.
"""

from __future__ import annotations

import math

import pytest

from cdm.config import DEFAULT_THRESHOLD, ROLLING_WINDOW
from cdm.drift import DriftEngine
from cdm.embeddings import cosine_distance
from cdm.models import DriftScore

# Orthonormal-ish basis vectors used throughout.
E0 = [1.0, 0.0, 0.0]
E1 = [0.0, 1.0, 0.0]
NEG_E0 = [-1.0, 0.0, 0.0]


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
# score_turn
# --------------------------------------------------------------------------- #
def test_score_turn_returns_real_driftscore() -> None:
    engine = DriftEngine()
    score = engine.score_turn(E0, E0, E1, turn_id=3, session_id="s1")
    assert isinstance(score, DriftScore)
    assert score.turn_id == 3
    assert score.session_id == "s1"


def test_score_turn_aligned_message_not_high() -> None:
    """A message identical to anchor and reference has zero drift, not flagged."""
    engine = DriftEngine(threshold=0.5)
    score = engine.score_turn(E0, E0, E0, turn_id=0, session_id="s")
    assert _approx(score.drift_from_anchor, 0.0)
    assert _approx(score.drift_from_reference, 0.0)
    assert score.is_drift_high is False


def test_score_turn_flags_when_anchor_distance_exceeds_threshold() -> None:
    """Orthogonal to anchor (distance 1.0) but aligned to reference still flags,
    because EITHER distance over threshold trips the flag."""
    engine = DriftEngine(threshold=0.5)
    # message E1: orthogonal to anchor E0 (distance 1.0), identical to reference E1 (0.0)
    score = engine.score_turn(E1, E0, E1, turn_id=1, session_id="s")
    assert _approx(score.drift_from_anchor, 1.0)
    assert _approx(score.drift_from_reference, 0.0)
    assert score.is_drift_high is True


def test_score_turn_flags_when_reference_distance_exceeds_threshold() -> None:
    """Aligned to anchor but far from reference -> flagged."""
    engine = DriftEngine(threshold=0.5)
    score = engine.score_turn(E0, E0, E1, turn_id=2, session_id="s")
    assert _approx(score.drift_from_anchor, 0.0)
    assert _approx(score.drift_from_reference, 1.0)
    assert score.is_drift_high is True


def test_score_turn_not_flagged_when_both_below_threshold() -> None:
    engine = DriftEngine(threshold=0.9)
    # distance 1.0 from anchor would NOT exceed 0.9? it does (1.0 > 0.9). Use identical.
    score = engine.score_turn(E0, E0, E0, turn_id=0, session_id="s")
    assert score.is_drift_high is False


def test_score_turn_threshold_is_strict_greater_than() -> None:
    """Distance exactly equal to the threshold must NOT flag (strict >)."""
    engine = DriftEngine(threshold=1.0)
    score = engine.score_turn(E1, E0, E1, turn_id=0, session_id="s")
    assert _approx(score.drift_from_anchor, 1.0)
    assert score.is_drift_high is False


def test_score_turn_none_reference_falls_back_to_anchor() -> None:
    engine = DriftEngine(threshold=0.5)
    score = engine.score_turn(E1, E0, None, turn_id=4, session_id="s")
    # reference distance should equal anchor distance exactly
    assert _approx(score.drift_from_reference, score.drift_from_anchor)
    assert _approx(score.drift_from_anchor, 1.0)
    assert score.is_drift_high is True


def test_score_turn_empty_reference_list_falls_back_to_anchor() -> None:
    """An empty reference embedding is falsy and must reuse the anchor distance."""
    engine = DriftEngine(threshold=0.5)
    score = engine.score_turn(E1, E0, [], turn_id=5, session_id="s")
    assert _approx(score.drift_from_reference, score.drift_from_anchor)


def test_score_turn_opposite_vectors_distance_two() -> None:
    engine = DriftEngine(threshold=1.5)
    score = engine.score_turn(NEG_E0, E0, None, turn_id=0, session_id="s")
    assert _approx(score.drift_from_anchor, 2.0)
    assert score.is_drift_high is True  # 2.0 > 1.5


# --------------------------------------------------------------------------- #
# centroid
# --------------------------------------------------------------------------- #
def test_centroid_empty_returns_empty() -> None:
    assert DriftEngine.centroid([]) == []


def test_centroid_single_vector_normalised() -> None:
    c = DriftEngine.centroid([[3.0, 4.0]])
    norm = math.sqrt(sum(x * x for x in c))
    assert _approx(norm, 1.0)
    assert _approx(c[0], 0.6)
    assert _approx(c[1], 0.8)


def test_centroid_is_unit_normalised() -> None:
    c = DriftEngine.centroid([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    norm = math.sqrt(sum(x * x for x in c))
    assert _approx(norm, 1.0)


def test_centroid_mean_direction() -> None:
    """Mean of two orthogonal unit vectors points along the diagonal."""
    c = DriftEngine.centroid([[1.0, 0.0], [0.0, 1.0]])
    assert _approx(c[0], c[1])
    assert _approx(c[0], 1.0 / math.sqrt(2.0))


def test_centroid_zero_mean_returns_empty() -> None:
    """Opposite vectors cancel; degenerate centroid -> []."""
    assert DriftEngine.centroid([[1.0, 0.0], [-1.0, 0.0]]) == []


# --------------------------------------------------------------------------- #
# rolling_reference
# --------------------------------------------------------------------------- #
def test_rolling_reference_empty() -> None:
    engine = DriftEngine(window=3)
    assert engine.rolling_reference([]) == []


def test_rolling_reference_uses_last_window_only() -> None:
    engine = DriftEngine(window=2)
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    # last 2 are both [0,1] -> centroid is [0,1]
    ref = engine.rolling_reference(embeddings)
    assert _approx(ref[0], 0.0)
    assert _approx(ref[1], 1.0)


def test_rolling_reference_fewer_than_window() -> None:
    engine = DriftEngine(window=10)
    ref = engine.rolling_reference([[1.0, 0.0]])
    assert _approx(ref[0], 1.0)
    assert _approx(ref[1], 0.0)


def test_rolling_reference_is_unit_normalised() -> None:
    engine = DriftEngine(window=ROLLING_WINDOW)
    ref = engine.rolling_reference([[2.0, 0.0], [0.0, 2.0]])
    norm = math.sqrt(sum(x * x for x in ref))
    assert _approx(norm, 1.0)


# --------------------------------------------------------------------------- #
# smooth
# --------------------------------------------------------------------------- #
def test_smooth_window_one_unchanged_copy() -> None:
    values = [1.0, 2.0, 3.0]
    out = DriftEngine.smooth(values, 1)
    assert out == values
    assert out is not values  # a copy, not the same object


def test_smooth_window_zero_unchanged() -> None:
    values = [5.0, 6.0]
    assert DriftEngine.smooth(values, 0) == values


def test_smooth_preserves_length() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert len(DriftEngine.smooth(values, 3)) == len(values)


def test_smooth_empty() -> None:
    assert DriftEngine.smooth([], 3) == []


def test_smooth_trailing_moving_average_values() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    out = DriftEngine.smooth(values, 2)
    # i=0: [1] -> 1.0
    # i=1: [1,2] -> 1.5
    # i=2: [2,3] -> 2.5
    # i=3: [3,4] -> 3.5
    assert _approx(out[0], 1.0)
    assert _approx(out[1], 1.5)
    assert _approx(out[2], 2.5)
    assert _approx(out[3], 3.5)


def test_smooth_window_larger_than_series() -> None:
    values = [2.0, 4.0]
    out = DriftEngine.smooth(values, 10)
    # i=0: [2] -> 2.0 ; i=1: [2,4] -> 3.0
    assert _approx(out[0], 2.0)
    assert _approx(out[1], 3.0)


def test_smooth_constant_series_unchanged_values() -> None:
    values = [7.0, 7.0, 7.0]
    out = DriftEngine.smooth(values, 3)
    assert all(_approx(v, 7.0) for v in out)


# --------------------------------------------------------------------------- #
# defaults
# --------------------------------------------------------------------------- #
def test_engine_defaults_from_config() -> None:
    engine = DriftEngine()
    assert engine.threshold == pytest.approx(DEFAULT_THRESHOLD)
    assert engine.window == ROLLING_WINDOW


def test_cosine_distance_sanity() -> None:
    """Guard the assumption the rest of the suite leans on."""
    assert cosine_distance(E0, E0) == pytest.approx(0.0, abs=1e-9)
    assert cosine_distance(E0, E1) == pytest.approx(1.0, abs=1e-9)
    assert cosine_distance(E0, NEG_E0) == pytest.approx(2.0, abs=1e-9)
