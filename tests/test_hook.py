"""Tests for the Claude Code drift hook (offline; synthetic transcripts)."""

from __future__ import annotations

import json

from cdm.hook import run_hook


class _StubEmbedder:
    """Deterministic 2-D embedder: on-goal text -> [1,0], off-goal -> [0,1].

    Lets the hook be tested end-to-end without the network or a real neural model.
    The fast hashing embedder can't separate on- from off-goal turns, so a useful
    "fires on real drift" test needs an embedder that actually distinguishes them.
    """

    name = "stub:2"
    dim = 2
    suggested_threshold = 0.5
    _OFF = ("banana", "bread", "recipe", "bake", "frosting", "football",
            "lunch", "snack", "taco", "movie", "weather")

    def encode_one(self, text):
        off = any(w in (text or "").lower() for w in self._OFF)
        return [0.0, 1.0] if off else [1.0, 0.0]

    def encode(self, texts):
        return [self.encode_one(t) for t in texts]


def _transcript(tmp_path, turns):
    p = tmp_path / "t.jsonl"
    with open(p, "w") as fh:
        for i, (role, text) in enumerate(turns):
            if role == "user":
                msg = {"role": "user", "content": text}
            else:
                msg = {"role": "assistant", "content": [{"type": "text", "text": text}]}
            fh.write(json.dumps({"type": role, "uuid": str(i), "message": msg}) + "\n")
    return str(p)


def test_hook_no_anchor_is_quiet(tmp_path):
    out = run_hook({"transcript_path": "/nonexistent/path.jsonl", "prompt": "hello there"})
    assert out["high"] is False
    assert out["context"] is None


def test_hook_quiet_until_enough_history(tmp_path):
    # The hook uses the self-calibrating verdict, which needs a few turns to learn a
    # baseline — a brand-new chat (too little history) is reported as on-track, even
    # for an off-topic prompt. (Sustained drift over a real baseline is what fires.)
    path = _transcript(tmp_path, [
        ("user", "build a rust cli that parses csv files"),
        ("assistant", "Sure, let's scaffold the rust CLI and a csv parser."),
    ])
    out = run_hook({"transcript_path": path, "prompt": "what's a good banana bread recipe"})
    assert out["high"] is False
    assert out["context"] is None


def test_hook_returns_documented_shape(tmp_path):
    path = _transcript(tmp_path, [("user", "a goal"), ("assistant", "ok")])
    out = run_hook({"transcript_path": path, "prompt": "a follow-up message"})
    assert {"high", "drift", "context"} <= set(out)
    assert isinstance(out["high"], bool) and isinstance(out["drift"], float)


def test_hook_flags_sustained_drift(tmp_path):
    # On-goal history, then a sustained topic change: the hook should fire once the
    # drift is established and surface a re-anchor pointing back at the goal.
    path = _transcript(tmp_path, [
        ("user", "build a rust cli that parses csv files"),
        ("assistant", "Sure, let's scaffold the rust CLI and a csv parser."),
        ("user", "add a flag to set the delimiter"),
        ("assistant", "Added the delimiter flag to the parser."),
        ("user", "handle quoted fields with commas"),
        ("assistant", "Quoted fields now parse in the rust csv parser."),
        ("user", "actually whats a good banana bread recipe"),
        ("assistant", "Here's a banana bread recipe to bake at home."),
        ("user", "how long do I bake the banana bread"),
    ])
    out = run_hook(
        {"transcript_path": path, "prompt": "what frosting goes on banana bread"},
        embedder=_StubEmbedder(),
    )
    assert out["high"] is True
    assert out["context"] and "rust" in out["context"].lower()


def test_hook_no_false_positive_on_divergent_ontopic(tmp_path):
    # Regression: a lexically-divergent but on-goal prompt must NOT be flagged.
    # The old raw per-turn threshold fired on every short on-goal prompt; the
    # self-calibrating verdict learns this conversation's own (flat) baseline and
    # stays quiet. Uses the real default hashing embedder, like the live hook.
    path = _transcript(tmp_path, [
        ("user", "build a rust cli that parses csv files"),
        ("assistant", "Sure, let's scaffold the rust CLI and a csv parser."),
        ("user", "add a flag to set the delimiter"),
        ("assistant", "Added the delimiter flag."),
        ("user", "handle quoted fields with commas"),
        ("assistant", "Quoted fields parse correctly now."),
    ])
    out = run_hook({"transcript_path": path, "prompt": "also support tab separated files"})
    assert out["high"] is False
    assert out["context"] is None
