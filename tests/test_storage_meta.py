"""Tests for the Store meta key/value layer (cross-process shared state)."""

from __future__ import annotations

from cdm.storage import Store


def test_meta_roundtrip(tmp_path):
    s = Store(tmp_path / "m.db")
    assert s.get_meta("missing") is None
    assert s.get_meta("missing", "fallback") == "fallback"

    s.set_meta("k", "v1")
    assert s.get_meta("k") == "v1"
    s.set_meta("k", "v2")  # upsert
    assert s.get_meta("k") == "v2"


def test_active_session_pointer(tmp_path):
    s = Store(tmp_path / "m.db")
    assert s.get_active_session() is None
    s.set_active_session("abc123")
    assert s.get_active_session() == "abc123"
    s.set_active_session(None)
    assert s.get_active_session() is None


def test_meta_visible_across_connections(tmp_path):
    """A second Store on the same file sees committed meta (WAL cross-process)."""
    db = tmp_path / "shared.db"
    writer = Store(db)
    writer.set_active_session("session-xyz")

    reader = Store(db)
    assert reader.get_active_session() == "session-xyz"
