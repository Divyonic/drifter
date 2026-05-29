"""Tests for cdm.watcher — clipboard auto-capture core logic.

Fully offline: a tmp-path store + the hashing embedder, and the testable
``poll_once``/``capture`` API driven with plain strings (no real clipboard, no
subprocess, no sleeping).
"""

from __future__ import annotations

import os

from cdm.corrective import CORRECTIVE_TEMPLATE
from cdm.embeddings import HashingEmbedder
from cdm.monitor import DriftMonitor
from cdm.storage import Store
from cdm.watcher import ClipboardWatcher, _pid_alive, is_watcher_running


def _watcher(tmp_path, **kwargs):
    store = Store(tmp_path / "w.db")
    monitor = DriftMonitor(store=store, embedder=HashingEmbedder(), threshold=0.8)
    return ClipboardWatcher(store=store, monitor=monitor, **kwargs), store, monitor


def test_no_active_session_no_capture(tmp_path):
    w, _store, _mon = _watcher(tmp_path)
    assert w.poll_once("a reasonably long sentence about gimbal mounts") is None


def test_capture_into_active_session(tmp_path):
    w, store, mon = _watcher(tmp_path)
    s = mon.start_session("p", "design a 5 kg pan-tilt gimbal mount", [])
    store.set_active_session(s.session_id)

    result = w.poll_once("we should size the stepper motors for the tilt axis")
    assert result is not None and "drift" in result
    msgs = store.get_messages(s.session_id)
    assert len(msgs) == 1
    assert msgs[0].role == "user"  # start_role default


def test_role_alternates_from_history(tmp_path):
    w, store, mon = _watcher(tmp_path)
    s = mon.start_session("p", "an anchor goal for the gimbal project", [])
    store.set_active_session(s.session_id)

    w.poll_once("first captured turn about the gimbal design please")
    w.poll_once("second different captured turn about motor torque values")
    w.poll_once("third distinct captured turn about bearing selection now")

    roles = [m.role for m in store.get_messages(s.session_id)]
    assert roles == ["user", "assistant", "user"]


def test_dedup_unchanged_clipboard(tmp_path):
    w, store, mon = _watcher(tmp_path)
    s = mon.start_session("p", "anchor goal text for the project here", [])
    store.set_active_session(s.session_id)

    text = "a captured conversation turn about the project goals"
    assert w.poll_once(text) is not None
    assert w.poll_once(text) is None  # identical clipboard -> not re-captured
    assert len(store.get_messages(s.session_id)) == 1


def test_skip_exact_repeat_of_last_message(tmp_path):
    w, store, mon = _watcher(tmp_path)
    s = mon.start_session("p", "anchor goal text for the project", [])
    store.set_active_session(s.session_id)

    text = "the first distinct captured turn about gimbal motors"
    assert w.poll_once(text) is not None
    # Clipboard "changes" (trailing space) but content equals the last stored turn.
    assert w.poll_once(text + " ") is None
    assert len(store.get_messages(s.session_id)) == 1


def test_should_capture_filters(tmp_path):
    w, _store, _mon = _watcher(tmp_path)
    assert w._should_capture("hi") is False                  # too short
    assert w._should_capture("supercalifragilistic") is False  # single word
    assert w._should_capture("this is a fine sentence") is True

    corrective = CORRECTIVE_TEMPLATE.format(
        core_goal="g",
        constraints="(none specified)",
        decisions="(none specified)",
        current_focus="f",
    )
    assert w._should_capture(corrective) is False  # never capture the corrective prompt


def test_capture_skips_deleted_active_session(tmp_path):
    w, store, mon = _watcher(tmp_path)
    s = mon.start_session("p", "anchor goal text for the project", [])
    store.set_active_session(s.session_id)
    store.delete_session(s.session_id)
    # Active pointer dangles -> capture is a no-op, no crash.
    assert w.poll_once("a captured turn that should be ignored entirely") is None


def test_pid_helpers_are_safe():
    assert _pid_alive(os.getpid()) is True
    assert _pid_alive(2_147_483_600) is False  # implausible pid
    assert isinstance(is_watcher_running(), bool)
