"""Tests for cdm.smart — LLM drift analysis (parsing + flow; no network/keys)."""

from __future__ import annotations

import cdm.smart as smart
from cdm.smart import _extract_json, analyze, build_user_prompt, parse_analysis


def test_parse_sub_task_is_not_drift():
    raw = ('{"core_goal":"build app","sub_goals":["auth","ui"],"current_focus":"css",'
           '"constraints":["x"],"status":"sub_task","drift":0.1,"reason":"narrow part","corrective":null}')
    v = parse_analysis(raw, "build app")
    assert v["status"] == "sub_task"
    assert v["drift"] == 0.1
    assert v["is_drift_high"] is False
    assert v["corrective"] is None
    assert v["sub_goals"] == ["auth", "ui"]


def test_parse_drifting_keeps_corrective():
    raw = '{"status":"drifting","drift":0.9,"reason":"off","corrective":"refocus please","core_goal":"g"}'
    v = parse_analysis(raw, "g")
    assert v["is_drift_high"] is True
    assert v["corrective"] == "refocus please"


def test_parse_tolerates_trailing_prose_and_fences():
    assert parse_analysis('{"status":"on_track","drift":0.0} thanks!', "g")["status"] == "on_track"
    assert parse_analysis('```json\n{"status":"evolved","drift":0.2}\n```', "g")["status"] == "evolved"


def test_parse_bad_status_and_missing_drift_defaults():
    assert parse_analysis('{"status":"weird"}', "g")["status"] == "on_track"
    assert parse_analysis('{"status":"drifting"}', "g")["drift"] > 0.5  # default high for drifting


def test_corrective_dropped_when_not_drifting():
    assert parse_analysis('{"status":"sub_task","drift":0.1,"corrective":"x"}', "g")["corrective"] is None


def test_extract_first_balanced_object():
    assert _extract_json('prefix {"a":1} {"b":2}') == {"a": 1}


def test_build_user_prompt_includes_goal_and_turns():
    p = build_user_prompt("my goal", [{"role": "user", "text": "hello"}, {"role": "assistant", "text": "hi"}])
    assert "my goal" in p and "User: hello" in p and "Assistant: hi" in p


def test_analyze_uses_client(monkeypatch):
    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def chat(self, messages, system=None):
            return '{"status":"sub_task","drift":0.1,"core_goal":"g"}'

    monkeypatch.setattr(smart, "LLMClient", FakeClient)
    v = analyze("g", [{"role": "user", "text": "x"}], provider="claude-cli")
    assert v["status"] == "sub_task" and v["drift"] == 0.1


def test_analyze_retries_once_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class FlakyClient:
        def __init__(self, *a, **k):
            pass

        def chat(self, messages, system=None):
            calls["n"] += 1
            return "no json at all" if calls["n"] == 1 else '{"status":"on_track","drift":0.0}'

    monkeypatch.setattr(smart, "LLMClient", FlakyClient)
    v = analyze("g", [{"role": "user", "text": "x"}], provider="claude-cli")
    assert v["status"] == "on_track" and calls["n"] == 2
