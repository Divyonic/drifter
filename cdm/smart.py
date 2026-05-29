"""LLM-based 'smart' drift analysis.

The offline engine measures drift as embedding distance — which is blind to intent:
deep work on a *sub-part* of a large goal looks like drift, and a goal that has
*legitimately evolved* gets flagged. Smart mode asks the connected LLM to read the
conversation and judge it properly:

- ``on_track``  — directly advancing the goal
- ``sub_task``  — narrowly focused on a legitimate part of a big goal (NOT drift)
- ``evolved``   — the goal has intentionally changed/expanded (update the anchor)
- ``drifting``  — genuinely off the goal

It returns a structured verdict (hierarchical goal, sub-goals, current focus,
constraints, a drift score, a one-line reason, and a paste-ready corrective only when
truly drifting). The LLM call is factored from the JSON parsing so parsing is
unit-testable offline; the call uses whatever provider you've connected (your Claude
subscription works, no API key).
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from cdm.llm import LLMClient

__all__ = ["SMART_SYSTEM", "build_user_prompt", "parse_analysis", "analyze", "STATUSES"]

STATUSES = ("on_track", "sub_task", "evolved", "drifting")

SMART_SYSTEM = """You are a context-drift analyst. Given a user's ORIGINAL GOAL and the recent CONVERSATION, judge whether the conversation is still serving that goal.

Be careful — these are NOT drift:
- Working narrowly on a sub-part of a large goal (label it "sub_task").
- The goal legitimately changing or expanding because the user chose to (label it "evolved", and update core_goal to reflect it).
Only label "drifting" when the conversation has genuinely left the goal: off-topic tangents, ignoring stated constraints, or losing the thread.

Return ONLY a JSON object (no prose, no markdown fences) with exactly these keys:
  "core_goal": string,            // the current main goal, refined if it evolved
  "sub_goals": [string],          // the parts/steps of the goal
  "current_focus": string,        // what is being worked on right now
  "constraints": [string],        // hard requirements you can infer
  "status": "on_track" | "sub_task" | "evolved" | "drifting",
  "drift": number,                // 0.0 = perfectly on goal, 1.0 = totally off; keep LOW for sub_task/evolved
  "reason": string,               // one short sentence
  "corrective": string or null    // a first-person re-anchor the user could paste; ONLY when status is "drifting", else null
"""


def build_user_prompt(anchor_goal: str, turns: List[dict], max_turns: int = 24) -> str:
    """Render the analysis request from the goal and the most-recent turns."""
    recent = turns[-max_turns:] if max_turns > 0 else turns
    lines = [f"ORIGINAL GOAL: {anchor_goal}".strip(), "", "CONVERSATION (most recent last):"]
    for t in recent:
        who = "User" if str(t.get("role")) == "user" else "Assistant"
        text = str(t.get("text", "")).strip().replace("\n", " ")
        if len(text) > 400:
            text = text[:400] + "…"
        if text:
            lines.append(f"{who}: {text}")
    lines.append("")
    lines.append("Return the JSON object only.")
    return "\n".join(lines)


def _first_balanced_object(s: str) -> Optional[str]:
    """Return the first balanced ``{...}`` object, respecting strings/escapes."""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of model output (tolerates fences + prose)."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fence.group(1) if fence else text
    try:
        return json.loads(candidate)
    except Exception:
        pass
    obj = _first_balanced_object(candidate) or _first_balanced_object(text)
    if obj is None:
        raise ValueError("no JSON object found in model output")
    return json.loads(obj)


def parse_analysis(raw_text: str, anchor_goal: str) -> Dict:
    """Parse + normalise the model's JSON into a stable verdict dict (pure)."""
    data = _extract_json(raw_text)
    status = str(data.get("status", "")).strip().lower()
    if status not in STATUSES:
        status = "on_track"
    try:
        drift = float(data.get("drift"))
    except (TypeError, ValueError):
        drift = 0.8 if status == "drifting" else 0.2
    drift = max(0.0, min(1.0, drift))

    def _strlist(v):
        return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []

    corrective = data.get("corrective")
    corrective = str(corrective).strip() if (corrective and status == "drifting") else None
    return {
        "core_goal": str(data.get("core_goal") or anchor_goal).strip(),
        "sub_goals": _strlist(data.get("sub_goals"))[:8],
        "current_focus": str(data.get("current_focus") or "").strip(),
        "constraints": _strlist(data.get("constraints"))[:10],
        "status": status,
        "drift": drift,
        "reason": str(data.get("reason") or "").strip(),
        "corrective": corrective,
        "is_drift_high": status == "drifting",
    }


def analyze(anchor_goal: str, turns: List[dict], provider: str, model: Optional[str] = None) -> Dict:
    """Run smart analysis via the connected LLM and return a normalised verdict.

    Raises :class:`cdm.llm.LLMError` on provider/SDK/credential problems, or
    ``ValueError`` if the model returns unparseable output.
    """
    client = LLMClient(provider, model=model, max_tokens=800)
    user_prompt = build_user_prompt(anchor_goal, turns)
    last_err: Optional[Exception] = None
    for attempt in range(2):
        system = SMART_SYSTEM if attempt == 0 else (
            SMART_SYSTEM + "\n\nIMPORTANT: Output ONLY the JSON object — no prose, no code fences."
        )
        raw = client.chat([{"role": "user", "content": user_prompt}], system=system)
        try:
            return parse_analysis(raw, anchor_goal)
        except Exception as exc:  # unparseable — retry once, then give up
            last_err = exc
    raise ValueError(f"smart analysis could not be parsed: {last_err}")
