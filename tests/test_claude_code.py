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
