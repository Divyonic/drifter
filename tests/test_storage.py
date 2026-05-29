"""Tests for cdm.storage.Store. Run fully offline (no embedder needed)."""

from __future__ import annotations

from cdm.models import DriftScore, GoalState, Message, Session
from cdm.storage import Store


def _make_store(tmp_path) -> Store:
    return Store(tmp_path / "sub" / "cdm.db")


def _session(session_id: str = "s1") -> Session:
    return Session(
        session_id=session_id,
        project_name="EO/IR Mount",
        anchor_goal="design a pan-tilt EO/IR mount under 5 kg",
        constraints=["< 5 kg", "must be sealed"],
        created_at="2026-05-29T00:00:00",
        updated_at="2026-05-29T00:00:00",
    )


def test_store_creates_parent_dir_and_tables(tmp_path):
    store = _make_store(tmp_path)
    assert (tmp_path / "sub" / "cdm.db").exists()
    # No sessions yet.
    assert store.list_sessions() == []


def test_session_round_trip(tmp_path):
    store = _make_store(tmp_path)
    sess = _session()
    store.create_session(sess)

    fetched = store.get_session("s1")
    assert fetched is not None
    assert fetched == sess
    assert fetched.constraints == ["< 5 kg", "must be sealed"]


def test_get_session_missing_returns_none(tmp_path):
    store = _make_store(tmp_path)
    assert store.get_session("nope") is None


def test_update_session(tmp_path):
    store = _make_store(tmp_path)
    sess = _session()
    store.create_session(sess)

    sess.project_name = "Renamed"
    sess.constraints = ["< 3 kg"]
    sess.updated_at = "2026-05-29T01:00:00"
    store.update_session(sess)

    fetched = store.get_session("s1")
    assert fetched is not None
    assert fetched.project_name == "Renamed"
    assert fetched.constraints == ["< 3 kg"]
    assert fetched.updated_at == "2026-05-29T01:00:00"


def test_list_sessions_newest_updated_first(tmp_path):
    store = _make_store(tmp_path)
    a = _session("a")
    a.updated_at = "2026-05-29T00:00:00"
    b = _session("b")
    b.updated_at = "2026-05-29T05:00:00"
    c = _session("c")
    c.updated_at = "2026-05-29T02:00:00"
    store.create_session(a)
    store.create_session(b)
    store.create_session(c)

    listed = store.list_sessions()
    assert [s.session_id for s in listed] == ["b", "c", "a"]


def test_message_round_trip_and_ordering(tmp_path):
    store = _make_store(tmp_path)
    store.create_session(_session())

    m0 = Message(
        turn_id=0,
        session_id="s1",
        role="user",
        text="Let's design the mount.",
        embedding=[0.1, 0.2, 0.3],
        created_at="2026-05-29T00:00:01",
    )
    m1 = Message(
        turn_id=1,
        session_id="s1",
        role="assistant",
        text="Sure, what is the payload?",
        embedding=None,
        created_at="2026-05-29T00:00:02",
    )
    # Insert out of order to confirm ordering by turn_id.
    store.add_message(m1)
    store.add_message(m0)

    msgs = store.get_messages("s1")
    assert [m.turn_id for m in msgs] == [0, 1]
    assert msgs[0] == m0
    assert msgs[0].embedding == [0.1, 0.2, 0.3]
    assert msgs[1].embedding is None


def test_next_turn_id_increments(tmp_path):
    store = _make_store(tmp_path)
    store.create_session(_session())

    assert store.next_turn_id("s1") == 0
    store.add_message(
        Message(turn_id=0, session_id="s1", role="user", text="a", created_at="t")
    )
    assert store.next_turn_id("s1") == 1
    store.add_message(
        Message(turn_id=1, session_id="s1", role="assistant", text="b", created_at="t")
    )
    assert store.next_turn_id("s1") == 2
    # Unknown session has no turns.
    assert store.next_turn_id("other") == 0


def test_goal_state_round_trip_and_latest(tmp_path):
    store = _make_store(tmp_path)
    store.create_session(_session())

    raw0 = {
        "core_goal": "design a pan-tilt EO/IR mount under 5 kg",
        "constraints": ["< 5 kg"],
        "decisions": [],
        "current_focus": "payload selection",
    }
    gs0 = GoalState(
        session_id="s1",
        turn_snapshot=-1,
        raw=raw0,
        reference_embedding=[0.5, 0.5],
        created_at="2026-05-29T00:00:00",
    )
    raw1 = {
        "core_goal": "design a pan-tilt EO/IR mount under 5 kg",
        "constraints": ["< 5 kg", "sealed"],
        "decisions": ["chose carbon fiber yoke"],
        "current_focus": "materials",
    }
    gs1 = GoalState(
        session_id="s1",
        turn_snapshot=4,
        raw=raw1,
        reference_embedding=None,
        created_at="2026-05-29T00:10:00",
    )
    store.add_goal_state(gs0)
    store.add_goal_state(gs1)

    all_states = store.get_goal_states("s1")
    assert [g.turn_snapshot for g in all_states] == [-1, 4]
    assert all_states[0] == gs0
    assert all_states[0].raw == raw0
    assert all_states[0].reference_embedding == [0.5, 0.5]

    latest = store.get_latest_goal_state("s1")
    assert latest is not None
    assert latest.turn_snapshot == 4
    assert latest.raw["decisions"] == ["chose carbon fiber yoke"]
    assert latest.reference_embedding is None


def test_get_latest_goal_state_missing_returns_none(tmp_path):
    store = _make_store(tmp_path)
    assert store.get_latest_goal_state("s1") is None


def test_drift_score_round_trip_and_ordering(tmp_path):
    store = _make_store(tmp_path)
    store.create_session(_session())

    d0 = DriftScore(
        turn_id=0,
        session_id="s1",
        drift_from_reference=0.1,
        drift_from_anchor=0.12,
        is_drift_high=False,
    )
    d2 = DriftScore(
        turn_id=2,
        session_id="s1",
        drift_from_reference=0.7,
        drift_from_anchor=0.8,
        is_drift_high=True,
    )
    store.add_drift_score(d2)
    store.add_drift_score(d0)

    scores = store.get_drift_scores("s1")
    assert [s.turn_id for s in scores] == [0, 2]
    assert scores[0] == d0
    assert scores[1].is_drift_high is True


def test_drift_score_upsert_replaces(tmp_path):
    store = _make_store(tmp_path)
    store.create_session(_session())

    store.add_drift_score(
        DriftScore(
            turn_id=3,
            session_id="s1",
            drift_from_reference=0.2,
            drift_from_anchor=0.2,
            is_drift_high=False,
        )
    )
    store.add_drift_score(
        DriftScore(
            turn_id=3,
            session_id="s1",
            drift_from_reference=0.9,
            drift_from_anchor=0.95,
            is_drift_high=True,
        )
    )

    scores = store.get_drift_scores("s1")
    assert len(scores) == 1
    assert scores[0].drift_from_reference == 0.9
    assert scores[0].drift_from_anchor == 0.95
    assert scores[0].is_drift_high is True


def test_delete_session_cascades(tmp_path):
    store = _make_store(tmp_path)
    store.create_session(_session())
    store.add_message(
        Message(turn_id=0, session_id="s1", role="user", text="hi", created_at="t")
    )
    store.add_goal_state(
        GoalState(session_id="s1", turn_snapshot=-1, raw={"core_goal": "x"})
    )
    store.add_drift_score(
        DriftScore(
            turn_id=0,
            session_id="s1",
            drift_from_reference=0.1,
            drift_from_anchor=0.1,
            is_drift_high=False,
        )
    )

    store.delete_session("s1")

    assert store.get_session("s1") is None
    assert store.get_messages("s1") == []
    assert store.get_goal_states("s1") == []
    assert store.get_drift_scores("s1") == []


def test_delete_session_isolates_other_sessions(tmp_path):
    store = _make_store(tmp_path)
    store.create_session(_session("keep"))
    store.create_session(_session("drop"))
    store.add_message(
        Message(turn_id=0, session_id="keep", role="user", text="hi", created_at="t")
    )
    store.add_message(
        Message(turn_id=0, session_id="drop", role="user", text="bye", created_at="t")
    )

    store.delete_session("drop")

    assert store.get_session("keep") is not None
    assert len(store.get_messages("keep")) == 1
    assert store.get_session("drop") is None
