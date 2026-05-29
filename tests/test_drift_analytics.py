"""Tests for the self-calibrating drift analytics (baseline/changepoint/forecast)."""

from __future__ import annotations

from cdm.drift import baseline_stats, biggest_jump, cusum_changepoint, forecast_cross
from cdm.embeddings import HashingEmbedder
from cdm.monitor import DriftMonitor
from cdm.storage import Store


def test_baseline_skips_leading_zero():
    mean, std = baseline_stats([0.0, 0.20, 0.22, 0.21], k=3)
    assert 0.18 < mean < 0.25
    assert std >= 0.03  # floored


def test_cusum_detects_sustained_shift_and_ignores_flat():
    series = [0.20, 0.21, 0.19, 0.20] + [0.9] * 5
    mean, std = baseline_stats(series, k=3)
    cp = cusum_changepoint(series, mean, std)
    assert cp is not None and cp >= 4  # fires in the shifted region

    flat = [0.2] * 8
    m, s = baseline_stats(flat, k=3)
    assert cusum_changepoint(flat, m, s) is None


def test_forecast_cross_cases():
    assert forecast_cross([0.2, 0.3, 0.4, 0.5, 0.6], 0.8) > 0  # rising -> turns ahead
    assert forecast_cross([0.9, 0.9], 0.8) == 0.0              # already above
    assert forecast_cross([0.5, 0.5, 0.5], 0.8) is None        # flat
    assert forecast_cross([0.6, 0.5, 0.4], 0.8) is None        # declining


def test_biggest_jump():
    assert biggest_jump([0.1, 0.12, 0.7, 0.72]) == 2
    assert biggest_jump([0.5, 0.4, 0.3]) is None


def test_monitor_timeseries_has_analytics(tmp_path):
    mon = DriftMonitor(store=Store(tmp_path / "a.db"), embedder=HashingEmbedder(), threshold=0.8)
    s = mon.start_session("p", "build a rust cli to parse csv files", [])
    turns = []
    for u in ["add a delimiter flag", "handle quoted fields in the parser"]:
        turns += [{"role": "user", "text": u}, {"role": "assistant", "text": "ok " + u}]
    for u in ["a good banana bread recipe please", "plan a weekend hiking trip"]:
        turns += [{"role": "user", "text": u}, {"role": "assistant", "text": "sure " + u}]
    mon.ingest_transcript(s.session_id, turns)
    ts = mon.timeseries(s.session_id)
    for key in ("baseline_mean", "baseline_std", "changepoint_turn",
                "attribution_turn", "forecast_turns", "forecast_will_cross"):
        assert key in ts
    assert ts["baseline_std"] >= 0.03
    assert isinstance(ts["forecast_will_cross"], bool)
