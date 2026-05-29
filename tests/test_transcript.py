"""Tests for cdm.transcript.parse_transcript (offline, no network)."""

from __future__ import annotations

import json

from cdm.transcript import parse_transcript


# --- JSON: bare list ---------------------------------------------------------

def test_json_list_of_role_content():
    src = json.dumps(
        [
            {"role": "user", "content": "Design a pan-tilt mount under 5 kg."},
            {"role": "assistant", "content": "Sure, here is a concept."},
        ]
    )
    out = parse_transcript(src)
    assert out == [
        {"role": "user", "text": "Design a pan-tilt mount under 5 kg."},
        {"role": "assistant", "text": "Sure, here is a concept."},
    ]


def test_json_role_synonyms_mapped():
    src = json.dumps(
        [
            {"speaker": "human", "text": "hi"},
            {"speaker": "ai", "text": "hello"},
            {"speaker": "bot", "text": "still here"},
            {"speaker": "model", "text": "and me"},
        ]
    )
    out = parse_transcript(src, fmt="json")
    assert [e["role"] for e in out] == ["user", "assistant", "assistant", "assistant"]


def test_json_unknown_role_defaults_to_user():
    src = json.dumps([{"role": "moderator", "content": "order in the court"}])
    out = parse_transcript(src)
    assert out == [{"role": "user", "text": "order in the court"}]


# --- JSON: object with messages / conversation key --------------------------

def test_json_messages_object():
    src = json.dumps(
        {
            "title": "session",
            "messages": [
                {"role": "user", "message": "first"},
                {"role": "assistant", "message": "second"},
            ],
        }
    )
    out = parse_transcript(src)
    assert out == [
        {"role": "user", "text": "first"},
        {"role": "assistant", "text": "second"},
    ]


def test_json_conversation_object():
    src = json.dumps(
        {"conversation": [{"author": "user", "value": "a"}, {"author": "assistant", "value": "b"}]}
    )
    out = parse_transcript(src)
    assert [e["text"] for e in out] == ["a", "b"]


def test_json_content_as_list_of_parts():
    src = json.dumps(
        [{"role": "assistant", "content": [{"type": "text", "text": "part1"}, {"text": "part2"}]}]
    )
    out = parse_transcript(src)
    assert out[0]["role"] == "assistant"
    assert "part1" in out[0]["text"] and "part2" in out[0]["text"]


def test_json_empty_text_turns_dropped():
    src = json.dumps([{"role": "user", "content": "   "}, {"role": "assistant", "content": "ok"}])
    out = parse_transcript(src)
    assert out == [{"role": "assistant", "text": "ok"}]


# --- Markdown / text speaker markers ----------------------------------------

def test_markdown_basic_markers():
    src = "User: hello there\nAssistant: hi, how can I help?"
    out = parse_transcript(src, fmt="markdown")
    assert out == [
        {"role": "user", "text": "hello there"},
        {"role": "assistant", "text": "hi, how can I help?"},
    ]


def test_markdown_heading_markers_and_multiline():
    src = (
        "## User\n"
        "Build a quadcopter.\n"
        "Budget is tight.\n"
        "## Assistant\n"
        "Understood.\n"
        "Here are options.\n"
    )
    out = parse_transcript(src)
    assert out == [
        {"role": "user", "text": "Build a quadcopter.\nBudget is tight."},
        {"role": "assistant", "text": "Understood.\nHere are options."},
    ]


def test_human_and_ai_markers_case_insensitive():
    src = "HUMAN: question one\nai: answer one\nHuman: question two"
    out = parse_transcript(src, fmt="text")
    assert [e["role"] for e in out] == ["user", "assistant", "user"]
    assert out[0]["text"] == "question one"


def test_markers_auto_detected():
    src = "User: what is the goal\nAssistant: keep it under 5 kg"
    out = parse_transcript(src)  # auto
    assert [e["role"] for e in out] == ["user", "assistant"]


# --- Marker-less alternation -------------------------------------------------

def test_no_markers_alternates_starting_user():
    src = "first line\nsecond line\nthird line\nfourth line"
    out = parse_transcript(src, fmt="text")
    assert [e["role"] for e in out] == ["user", "assistant", "user", "assistant"]
    assert [e["text"] for e in out] == ["first line", "second line", "third line", "fourth line"]


def test_no_markers_blank_lines_skipped():
    src = "alpha\n\n\nbeta\n"
    out = parse_transcript(src, fmt="markdown")
    assert [e["text"] for e in out] == ["alpha", "beta"]
    assert [e["role"] for e in out] == ["user", "assistant"]


# --- Garbage / edge cases (never raise) -------------------------------------

def test_empty_string_returns_empty_list():
    assert parse_transcript("") == []
    assert parse_transcript("   \n  \n ") == []


def test_garbage_json_falls_back_or_empty():
    # Broken JSON-ish text with no markers -> alternation best effort, never raises.
    out = parse_transcript('{"messages": [ broken', fmt="auto")
    assert isinstance(out, list)
    for e in out:
        assert set(e.keys()) == {"role", "text"}
        assert e["role"] in ("user", "assistant")


def test_pure_garbage_does_not_raise():
    out = parse_transcript("\x00\x01\x02 ~~~ !!! ???")
    assert isinstance(out, list)


def test_non_string_source_does_not_raise():
    assert parse_transcript(None) == []  # type: ignore[arg-type]
    assert isinstance(parse_transcript(12345), list)  # type: ignore[arg-type]


def test_json_empty_list_returns_empty():
    assert parse_transcript("[]", fmt="json") == []


def test_json_single_object_message():
    src = json.dumps({"role": "user", "content": "lonely message"})
    out = parse_transcript(src)
    assert out == [{"role": "user", "text": "lonely message"}]


# --- File path source --------------------------------------------------------

def test_source_as_file_path(tmp_path):
    p = tmp_path / "convo.json"
    p.write_text(
        json.dumps([{"role": "user", "content": "from a file"}, {"role": "ai", "content": "ack"}]),
        encoding="utf-8",
    )
    out = parse_transcript(str(p))
    assert out == [
        {"role": "user", "text": "from a file"},
        {"role": "assistant", "text": "ack"},
    ]


def test_source_as_markdown_file(tmp_path):
    p = tmp_path / "convo.md"
    p.write_text("User: hi\nAssistant: yo", encoding="utf-8")
    out = parse_transcript(str(p))
    assert [e["role"] for e in out] == ["user", "assistant"]


def test_nonexistent_path_treated_as_text():
    # Looks like a path but does not exist -> parsed as raw text (alternation).
    out = parse_transcript("/no/such/file/here.txt")
    assert isinstance(out, list)


# --- JSON list of bare strings ----------------------------------------------

def test_json_list_of_strings_alternates():
    src = json.dumps(["q1", "a1", "q2"])
    out = parse_transcript(src)
    assert [e["role"] for e in out] == ["user", "assistant", "user"]
    assert [e["text"] for e in out] == ["q1", "a1", "q2"]
