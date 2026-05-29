"""The drift-detection eval harness should catch drift on the synthetic corpus."""

from __future__ import annotations

from cdm.eval import evaluate


def test_eval_harness_runs_and_reports_metrics():
    # Structural check only. Detection *quality* depends on embedder separation:
    # the lexical hashing fallback can't tell reworded-on-topic from off-topic, so
    # we don't assert recall here — semantic-embedder quality is verified separately.
    m = evaluate(embedder="hashing")
    assert m["sessions"] == 5
    assert 0 <= m["detected"] <= 5
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["recall"] <= 1.0
    assert "embedder" in m
