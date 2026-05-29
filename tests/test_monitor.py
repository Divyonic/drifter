"""Tests for cdm.monitor.DriftMonitor.

Fully offline: a tmp-path SQLite store plus the deterministic
:class:`~cdm.embeddings.HashingEmbedder` (no network, no API keys, no
sentence-transformers). The scenario starts on a clear engineering goal and then
visibly drifts into unrelated office chatter, so drift should rise and a
corrective prompt should appear.
"""

from __future__ import annotations

import pytest

from cdm.config import DEFAULT_THRESHOLD
from cdm.embeddings import HashingEmbedder
from cdm.models import DriftScore, GoalState, Message, Session
from cdm.monitor import DriftMonitor
from cdm.storage import Store

ANCHOR = "design a pan-tilt EO/IR gimbal mount under 5 kg for a drone payload"
CONSTRAINTS = ["< 5 kg", "must be sealed"]

# On-topic opening followed by a clear drift into unrelated chatter.
ON_TOPIC = [
    ("user", "lets design the pan-tilt EO/IR gimbal mount for the drone payload"),
    ("assistant", "great, what payload and weight budget are we targeting for the mount"),
    ("user", "keep the gimbal mount under 5 kg and we decided to go with a carbon fiber yoke"),
    ("assistant", "carbon fiber yoke noted, sealed bearings for the pan-tilt axes"),
]
OFF_TOPIC = [
    ("user", "what snacks should we get for the office break room next week"),
    ("assistant", "maybe some chips and fruit and sparkling water for the office"),
    ("user", "i think we should plan a team lunch at the new taco place downtown"),
    ("assistant", "tacos sound great, friday works for the whole team lunch"),
    ("user", "did you watch the football game last night it was an amazing finish"),
    ("assistant", "yeah the football game was a wild overtime ending for sure"),
]


def _monitor(tmp_path) -> DriftMonitor:
    """Build a DriftMonitor on a tmp-path store with the offline hashing embedder."""
    store = Store(tmp_path / "cdm.db")
    return DriftMonitor(store=store, embedder=HashingEmbedder(), threshold=DEFAULT_THRESHOLD)


def test_start_session_persists_session_and_initial_goal_state(tmp_path):
    mon = _monitor(tmp_path)
    session = mon.start_session("EO/IR Mount", ANCHOR, CONSTRAINTS)

    assert isinstance(session, Session)
    assert session.session_id  # uuid4 hex, non-empty
    assert session.anchor_goal == ANCHOR
    assert session.constraints == CONSTRAINTS
    assert session.created_at and session.updated_at  # ISO timestamps set

    # Round-trips through storage.
    assert mon.store.get_session(session.session_id) == session

    # An initial goal state (turn_snapshot=-1) is persisted, anchored.
    gs = mon.latest_goal_state(session.session_id)
    assert isinstance(gs, GoalState)
    assert gs.turn_snapshot == -1
    assert gs.raw["core_goal"] == ANCHOR
    assert gs.reference_embedding is not None
    assert len(gs.reference_embedding) == mon.embedder.dim


def test_add_turn_returns_documented_shape(tmp_path):
    mon = _monitor(tmp_path)
    session = mon.start_session("EO/IR Mount", ANCHOR, CONSTRAINTS)

    result = mon.add_turn(session.session_id, "user", ON_TOPIC[0][1])

    assert set(result.keys()) == {
        "message",
        "drift",
        "goal_state",
        "alert",
        "corrective_prompt",
    }
    assert isinstance(result["message"], Message)
    assert result["message"].turn_id == 0
    assert result["message"].embedding is not None
    assert isinstance(result["drift"], DriftScore)
    assert isinstance(result["goal_state"], GoalState)
    assert isinstance(result["alert"], bool)
    # On-topic opening turn should NOT alert; prompt only set on alert.
    assert result["alert"] is False
    assert result["corrective_prompt"] is None


def test_add_turn_unknown_session_raises(tmp_path):
    mon = _monitor(tmp_path)
    with pytest.raises(ValueError):
        mon.add_turn("does-not-exist", "user", "hello")


def test_drift_rises_and_alert_with_corrective_prompt_appears(tmp_path):
    mon = _monitor(tmp_path)
    session = mon.start_session("EO/IR Mount", ANCHOR, CONSTRAINTS)

    drifts: list[float] = []
    saw_alert = False
    last_prompt = None
    for role, text in ON_TOPIC + OFF_TOPIC:
        result = mon.add_turn(session.session_id, role, text)
        drifts.append(result["drift"].drift_from_anchor)
        if result["alert"]:
            saw_alert = True
            # When an alert fires, a paste-ready corrective prompt is returned.
            assert isinstance(result["corrective_prompt"], str)
            assert result["corrective_prompt"]
            last_prompt = result["corrective_prompt"]

    # The opening on-topic turn stays under the threshold...
    assert drifts[0] < DEFAULT_THRESHOLD
    # ...and an alert eventually fires as the conversation drifts off-goal.
    assert saw_alert

    # Drift clearly rises from the on-topic opening to the off-topic tail.
    early = sum(drifts[: len(ON_TOPIC)]) / len(ON_TOPIC)
    late = sum(drifts[-len(OFF_TOPIC):]) / len(OFF_TOPIC)
    assert late > early
    # The off-topic tail is unambiguously above the threshold.
    assert late > DEFAULT_THRESHOLD

    # The latest off-topic turns are flagged high in the persisted scores.
    scores = mon.store.get_drift_scores(session.session_id)
    assert any(s.is_drift_high for s in scores[-len(OFF_TOPIC):])

    # The corrective prompt restates the (stable) anchor goal.
    assert last_prompt is not None
    assert ANCHOR in last_prompt
    # And current_corrective_prompt mirrors it from the latest goal state.
    assert ANCHOR in mon.current_corrective_prompt(session.session_id)


def test_timeseries_shape_and_smoothing(tmp_path):
    mon = _monitor(tmp_path)
    session = mon.start_session("EO/IR Mount", ANCHOR, CONSTRAINTS)
    for role, text in ON_TOPIC + OFF_TOPIC:
        mon.add_turn(session.session_id, role, text)

    ts = mon.timeseries(session.session_id)
    n = len(ON_TOPIC) + len(OFF_TOPIC)

    assert {
        "turns",
        "roles",
        "texts",
        "drift_from_anchor",
        "drift_from_reference",
        "threshold",
        "alignment_events",
    } <= set(ts.keys())
    # All parallel series share the same length (one per stored turn).
    assert ts["turns"] == list(range(n))
    assert len(ts["roles"]) == n
    assert len(ts["texts"]) == n
    assert len(ts["drift_from_anchor"]) == n
    assert len(ts["drift_from_reference"]) == n
    assert ts["threshold"] == DEFAULT_THRESHOLD

    # Texts are trimmed strings; roles are the recorded roles.
    assert all(isinstance(t, str) for t in ts["texts"])
    assert ts["roles"][0] == "user"

    # alignment_events is a subset of turn ids, all flagged high.
    assert set(ts["alignment_events"]).issubset(set(ts["turns"]))
    assert ts["alignment_events"]  # at least one drift-high turn


def test_ingest_transcript_returns_documented_shape(tmp_path):
    mon = _monitor(tmp_path)
    session = mon.start_session("EO/IR Mount", ANCHOR, CONSTRAINTS)

    turns = [{"role": r, "text": t} for r, t in ON_TOPIC + OFF_TOPIC]
    # Blank/whitespace turns are skipped, not counted.
    turns.append({"role": "user", "text": "   "})

    result = mon.ingest_transcript(session.session_id, turns)

    assert set(result.keys()) == {"added", "alerts", "final_goal_state"}
    assert result["added"] == len(ON_TOPIC) + len(OFF_TOPIC)
    assert isinstance(result["alerts"], int)
    assert result["alerts"] >= 1  # the off-topic tail drifts high
    assert isinstance(result["final_goal_state"], GoalState)
    assert result["final_goal_state"].raw["core_goal"] == ANCHOR


def test_set_threshold_updates_engine_and_monitor(tmp_path):
    mon = _monitor(tmp_path)
    mon.set_threshold(0.9)
    assert mon.threshold == 0.9
    assert mon.engine.threshold == 0.9

    session = mon.start_session("EO/IR Mount", ANCHOR, CONSTRAINTS)
    # A turn that drifts ~0.95 from anchor is high at 0.9 but would not be at a
    # higher cutoff; confirm the new threshold flows into scoring + timeseries.
    result = mon.add_turn(session.session_id, "user", OFF_TOPIC[0][1])
    assert result["drift"].drift_from_anchor > 0.9
    assert result["alert"] is True
    assert mon.timeseries(session.session_id)["threshold"] == 0.9


def test_mark_checkpoint_adds_goal_state_revision(tmp_path):
    mon = _monitor(tmp_path)
    session = mon.start_session("EO/IR Mount", ANCHOR, CONSTRAINTS)
    for role, text in ON_TOPIC:
        mon.add_turn(session.session_id, role, text)

    before = len(mon.store.get_goal_states(session.session_id))
    mon.mark_checkpoint(session.session_id)
    after = len(mon.store.get_goal_states(session.session_id))

    assert after == before + 1
    latest = mon.latest_goal_state(session.session_id)
    assert isinstance(latest, GoalState)
    # Checkpoint is pinned to the last turn and keeps the stable anchor goal.
    assert latest.turn_snapshot == len(ON_TOPIC) - 1
    assert latest.raw["core_goal"] == ANCHOR


def test_latest_goal_state_none_for_unknown_session(tmp_path):
    mon = _monitor(tmp_path)
    assert mon.latest_goal_state("nope") is None
