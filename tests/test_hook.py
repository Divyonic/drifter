"""Tests for the Claude Code drift hook (offline; synthetic transcripts)."""

from __future__ import annotations

import json

from cdm.hook import run_hook


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


def test_hook_flags_offtopic_prompt(tmp_path):
    path = _transcript(tmp_path, [
        ("user", "build a rust cli that parses csv files"),
        ("assistant", "Sure, let's scaffold the rust CLI and a csv parser."),
    ])
    out = run_hook({"transcript_path": path, "prompt": "what's a good banana bread recipe"}, threshold=0.6)
    assert out["high"] is True
    assert out["context"] and "rust" in out["context"].lower()


def test_hook_quiet_when_ontopic(tmp_path):
    path = _transcript(tmp_path, [
        ("user", "build a rust cli that parses csv files"),
        ("assistant", "Sure, let's scaffold it."),
    ])
    out = run_hook(
        {"transcript_path": path, "prompt": "add a flag to set the csv delimiter for the rust cli parser"},
        threshold=0.98,
    )
    assert out["high"] is False
    assert out["context"] is None


def test_hook_no_anchor_is_quiet(tmp_path):
    out = run_hook({"transcript_path": "/nonexistent/path.jsonl", "prompt": "hello there"})
    assert out["high"] is False
    assert out["context"] is None
