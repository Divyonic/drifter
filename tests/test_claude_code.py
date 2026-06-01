"""Tests for cdm.claude_code — parsing & tailing Claude Code transcripts (offline)."""

from __future__ import annotations

import json

from cdm.claude_code import ClaudeCodeTail, extract_turn, parse_transcript_file


def _write(path, objs):
    with open(path, "w") as fh:
        for o in objs:
            fh.write(json.dumps(o) + "\n")


def test_extract_user_string():
    t = extract_turn({"type": "user", "uuid": "1", "message": {"role": "user", "content": "hello world"}})
    assert t and t["role"] == "user" and t["text"] == "hello world"


def test_extract_assistant_text_blocks_only():
    o = {"type": "assistant", "uuid": "2", "message": {"role": "assistant", "content": [
        {"type": "thinking", "text": "hmm"},
        {"type": "text", "text": "the answer"},
        {"type": "tool_use", "name": "x"},
    ]}}
    t = extract_turn(o)
    assert t and t["role"] == "assistant" and t["text"] == "the answer"


def test_skip_sidechain_meta_and_tool_only():
    assert extract_turn({"type": "user", "isSidechain": True, "uuid": "3",
                         "message": {"role": "user", "content": "x"}}) is None
    assert extract_turn({"type": "system", "uuid": "4"}) is None
    assert extract_turn({"type": "assistant", "uuid": "5",
                         "message": {"role": "assistant", "content": [{"type": "tool_use", "name": "x"}]}}) is None
    assert extract_turn({"type": "user", "uuid": "6",
                         "message": {"role": "user", "content": [{"type": "tool_result", "content": "r"}]}}) is None


def test_skip_command_noise():
    assert extract_turn({"type": "user", "uuid": "7",
                         "message": {"role": "user", "content": "<command-name>/foo</command-name>"}}) is None


def test_parse_orders_and_dedupes(tmp_path):
    p = tmp_path / "t.jsonl"
    _write(p, [
        {"type": "user", "uuid": "a", "message": {"role": "user", "content": "goal here"}},
        {"type": "assistant", "uuid": "b", "message": {"role": "assistant", "content": [{"type": "text", "text": "reply"}]}},
        {"type": "assistant", "uuid": "b", "message": {"role": "assistant", "content": [{"type": "text", "text": "dup"}]}},
        {"type": "system", "uuid": "c"},
    ])
    turns = parse_transcript_file(str(p))
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[0]["text"] == "goal here"


def test_tail_start_at_end_then_appends(tmp_path):
    p = tmp_path / "t.jsonl"
    _write(p, [{"type": "user", "uuid": "a", "message": {"role": "user", "content": "old"}}])
    tail = ClaudeCodeTail(str(p), start_at_end=True)
    assert tail.new_turns() == []
    with open(p, "a") as fh:
        fh.write(json.dumps({"type": "assistant", "uuid": "b",
                             "message": {"role": "assistant", "content": [{"type": "text", "text": "new reply"}]}}) + "\n")
    nt = tail.new_turns()
    assert len(nt) == 1 and nt[0]["text"] == "new reply"
    assert tail.new_turns() == []  # not re-emitted


def test_snapshot_and_find_new_transcript(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    from cdm import claude_code as cc

    proj = tmp_path / "projects" / "-Users-me-proj"
    proj.mkdir(parents=True)
    old = proj / "old.jsonl"
    _write(old, [{"type": "user", "uuid": "1", "cwd": "/Users/me/proj",
                  "message": {"role": "user", "content": "hi"}}])
    before = cc.snapshot_transcripts()
    assert str(old) in before

    new = proj / "new.jsonl"
    _write(new, [{"type": "user", "uuid": "2", "cwd": "/Users/me/proj",
                  "message": {"role": "user", "content": "goal"}}])
    assert cc.find_new_transcript(before, cwd="/Users/me/proj") == str(new)
    # nothing new relative to the current snapshot
    assert cc.find_new_transcript(cc.snapshot_transcripts()) is None


def test_launch_returns_false_without_tmux_off_mac(monkeypatch):
    from cdm import claude_code as cc

    monkeypatch.setattr(cc.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cc.shutil, "which", lambda *_a, **_k: None)  # no tmux, no claude
    assert cc.launch_claude_in_terminal("/tmp", "hello") == (False, None)
    # An applescript handle is a no-op off macOS.
    assert cc.send_to_terminal("applescript:/dev/ttys003", "hi there") is False


def test_tail_waits_for_complete_line(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("")
    tail = ClaudeCodeTail(str(p))
    with open(p, "a") as fh:  # partial line, no newline yet
        fh.write('{"type":"user","uuid":"a","message":{"role":"user","content":"partial"}}')
    assert tail.new_turns() == []
    with open(p, "a") as fh:
        fh.write("\n")
    nt = tail.new_turns()
    assert len(nt) == 1 and nt[0]["text"] == "partial"


def test_applescript_send_script_targets_tty_and_escapes():
    from cdm.claude_code import _applescript_send_script

    script = _applescript_send_script("/dev/ttys003", 'say "hi" now')
    assert 'tty of t is "/dev/ttys003"' in script
    # Double quotes in the message are escaped so the AppleScript stays valid.
    assert '\\"hi\\"' in script


def test_send_to_terminal_rejects_empty_inputs():
    from cdm.claude_code import send_to_terminal

    # No handle / blank text => never shells out, on any platform.
    assert send_to_terminal("", "hello") is False
    assert send_to_terminal("tmux:sess", "   ") is False
    assert send_to_terminal("unknown:thing", "hello") is False  # unknown backend


def test_launch_prefers_tmux(monkeypatch):
    """When tmux exists, it is used on any platform and yields a tmux: handle."""
    from cdm import claude_code as cc

    monkeypatch.setattr(cc.shutil, "which", lambda c, *_a, **_k: "/usr/bin/" + c)
    monkeypatch.setattr(cc, "_open_terminal_attached", lambda *_a, **_k: True)
    calls = []

    class _R:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(cc.subprocess, "run", lambda argv, **kw: calls.append(argv) or _R())

    ok, handle = cc.launch_claude_in_terminal("/work", "do the thing", anchor="A", name="proj")
    assert ok is True and handle.startswith("tmux:")
    assert calls and calls[0][:3] == ["tmux", "new-session", "-d"]


def test_send_to_terminal_tmux_uses_send_keys(monkeypatch):
    """tmux forwarding sends the collapsed text literally, then a separate Enter."""
    from cdm import claude_code as cc

    calls = []

    class _R:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(cc.subprocess, "run", lambda argv, **kw: calls.append(argv) or _R())

    assert cc.send_to_terminal("tmux:drifter_x", "hello\nworld  again") is True
    assert calls[0] == ["tmux", "send-keys", "-t", "drifter_x", "-l", "hello world again"]
    assert calls[1] == ["tmux", "send-keys", "-t", "drifter_x", "Enter"]
