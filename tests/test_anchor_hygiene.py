"""Tests for the anchor-selection + constraint-hygiene fixes.

These cover the bug surfaced when Drifter monitored a long, meta Claude Code
session and emitted a useless corrective:

  * it anchored on the literal first turn ("… delete this branch"), and
  * it mined transcript/log lines ("<failures>", a usage/token line,
    "i cant zeem zoom in") as if they were user constraints.

Fully offline and deterministic: standard library + the hashing embedder only.
"""

from __future__ import annotations

from cdm.embeddings import HashingEmbedder
from cdm.goal_state import _looks_like_noise, extract_goal_state
from cdm.models import Message
from cdm.monitor import DriftMonitor
from cdm.storage import Store
from cdm.transcript import pick_anchor_goal


def _msg(turn_id: int, role: str, text: str) -> Message:
    return Message(turn_id=turn_id, session_id="s1", role=role, text=text)


REAL_GOAL = (
    "declutter the monitor UI: make the sidebar collapsible, move theme into "
    "settings, and improve responsiveness so the chat fills the column"
)


# --- pick_anchor_goal --------------------------------------------------------

def test_anchor_skips_delete_branch_first_turn():
    """The exact misfire: turn 1 is a throwaway command, not the goal."""
    turns = [
        {"role": "user", "text": "cursor/defence-tech-interactivity-50f6 delete this branch"},
        {"role": "assistant", "text": "Deleted the branch."},
        {"role": "user", "text": REAL_GOAL},
        {"role": "assistant", "text": "On it — collapsing the sidebar first."},
    ]
    assert pick_anchor_goal(turns) == REAL_GOAL


def test_anchor_skips_branch_names_and_slash_commands():
    turns = [
        {"role": "user", "text": "feature/ui-cleanup-42a"},
        {"role": "user", "text": "/compact"},
        {"role": "user", "text": "git checkout main"},
        {"role": "user", "text": REAL_GOAL},
    ]
    assert pick_anchor_goal(turns) == REAL_GOAL


def test_anchor_skips_markup_lines():
    turns = [
        {"role": "user", "text": "<system-reminder>context loaded</system-reminder>"},
        {"role": "user", "text": REAL_GOAL},
    ]
    assert pick_anchor_goal(turns) == REAL_GOAL


def test_anchor_falls_back_to_most_substantial_when_nothing_goal_like():
    # Only command-shaped turns: pick the longest rather than blindly turn 0.
    turns = [
        {"role": "user", "text": "/clear"},
        {"role": "user", "text": "git commit -m 'wip' --amend --no-edit then push"},
    ]
    assert pick_anchor_goal(turns) == "git commit -m 'wip' --amend --no-edit then push"


def test_anchor_empty_and_fallback():
    assert pick_anchor_goal([]) == ""
    assert pick_anchor_goal([], fallback="x") == "x"


def test_anchor_prefers_user_over_assistant():
    turns = [
        {"role": "assistant", "text": "Here is a long helpful assistant preamble " * 5},
        {"role": "user", "text": REAL_GOAL},
    ]
    assert pick_anchor_goal(turns) == REAL_GOAL


# --- _looks_like_noise -------------------------------------------------------

def test_noise_detector_rejects_transcript_shapes():
    assert _looks_like_noise("<failures>3</failures>")
    assert _looks_like_noise("input_tokens: 152340")
    assert _looks_like_noise("cache_read: 0")
    assert _looks_like_noise("we used 150000 tokens in this run")
    assert _looks_like_noise("status: ok")
    assert _looks_like_noise("12345 / 67890 :: 0x9f == ----")  # code/ID-heavy, not prose
    # version-control / shell plumbing
    assert _looks_like_noise("cursor/defence-tech-interactivity-50f6 delete this branch")
    assert _looks_like_noise("Deleted the branch.")
    assert _looks_like_noise("merge branch into main")
    assert _looks_like_noise("/compact")


def test_noise_detector_keeps_real_prose():
    assert not _looks_like_noise("the gimbal must weigh under 5 kg")
    assert not _looks_like_noise("budget: stay under $200 for the whole build")
    assert not _looks_like_noise("we decided to go with a carbon fiber yoke")


# --- constraint hygiene via extract_goal_state -------------------------------

def test_log_lines_never_become_constraints():
    msgs = [
        _msg(0, "user", "the mount must weigh under 5 kg"),
        _msg(1, "assistant", "noted"),
        _msg(2, "user", "<failures>0</failures>"),
        _msg(3, "user", "input_tokens: 152340 output_tokens: 4096"),
        _msg(4, "user", "we used 150000 tokens so far"),
    ]
    raw = extract_goal_state(msgs, anchor_goal="design a gimbal mount",
                             anchor_constraints=[], turn_snapshot=4)
    cons = raw["constraints"]
    assert any("under 5 kg" in c.lower() for c in cons)  # the real one survives
    assert not any("token" in c.lower() for c in cons)
    assert not any("failures" in c.lower() for c in cons)


def test_bare_cant_is_not_a_constraint():
    """'i cant zeem zoom in' is a complaint, not a requirement."""
    msgs = [_msg(0, "user", "i cant zeem zoom in")]
    raw = extract_goal_state(msgs, anchor_goal="build the chart",
                             anchor_constraints=[], turn_snapshot=0)
    assert raw["constraints"] == []


def test_real_prohibition_still_captured():
    msgs = [_msg(0, "user", "the API key must not be stored remotely")]
    raw = extract_goal_state(msgs, anchor_goal="secure storage",
                             anchor_constraints=[], turn_snapshot=0)
    assert any("must not be stored remotely" in c.lower() for c in raw["constraints"])


# --- focus keyword hygiene ---------------------------------------------------

def test_focus_drops_generic_leak_words():
    msgs = [
        _msg(0, "user", "now add two gimbal things"),
        _msg(1, "user", "now add the gimbal mount and gimbal yoke"),
        _msg(2, "user", "gimbal gimbal now two"),
    ]
    raw = extract_goal_state(msgs, anchor_goal="g", anchor_constraints=[], turn_snapshot=2)
    focus = raw["current_focus"]
    assert "gimbal" in focus
    for junk in ("now", "add", "two"):
        assert junk not in focus.split(", ")


def test_focus_drops_transcript_noise_tokens():
    """Markup/usage/branch-slug tokens must not leak into focus keywords."""
    msgs = [
        _msg(0, "user", "cursor/defence-tech-interactivity-50f6 delete this branch"),
        _msg(1, "assistant", "<usage>input_tokens: 152340 output_tokens: 4096</usage>"),
        _msg(2, "user", "<failures>0</failures>"),
        _msg(3, "user", "make the sidebar collapsible and the chart frame to the drift band"),
    ]
    focus = extract_goal_state(
        msgs, anchor_goal="g", anchor_constraints=[], turn_snapshot=3
    )["current_focus"].split(", ")
    for junk in ("usage", "tokens", "failures", "defence-tech-interactivity",
                 "input_tokens", "branch", "cursor", "delete", "deleted"):
        assert junk not in focus
    assert any(w in focus for w in ("sidebar", "chart", "drift", "band", "collapsible"))


# --- monitor.set_goal (the mutable-anchor primitive) -------------------------

def _monitor(tmp_path) -> DriftMonitor:
    return DriftMonitor(store=Store(tmp_path / "cdm.db"), embedder=HashingEmbedder())


def test_set_goal_reanchors_and_reembeds(tmp_path):
    mon = _monitor(tmp_path)
    s = mon.start_session("Imported chat", "cursor/x-50f6 delete this branch", [])
    mon.add_turn(s.session_id, "user", "actually lets " + REAL_GOAL)
    bad_anchor_emb = list(mon._anchor_embedding(mon.store.get_session(s.session_id)))

    updated = mon.set_goal(s.session_id, REAL_GOAL)

    assert updated.anchor_goal == REAL_GOAL
    persisted = mon.store.get_session(s.session_id)
    assert persisted.anchor_goal == REAL_GOAL
    # Cache was invalidated and re-embedded to the new anchor.
    new_anchor_emb = mon._anchor_embedding(persisted)
    assert new_anchor_emb != bad_anchor_emb
    # The goal state now reflects the corrected goal.
    gs = mon.latest_goal_state(s.session_id)
    assert gs is not None and gs.raw["core_goal"] == REAL_GOAL
    # Timeseries still works after re-anchoring.
    assert mon.timeseries(s.session_id)["threshold"] > 0


def test_set_goal_rejects_empty_and_unknown(tmp_path):
    mon = _monitor(tmp_path)
    s = mon.start_session("p", "a real goal here", [])
    import pytest

    with pytest.raises(ValueError):
        mon.set_goal(s.session_id, "   ")
    with pytest.raises(ValueError):
        mon.set_goal("nope", REAL_GOAL)
