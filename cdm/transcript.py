"""Transcript parsing for the Context Drift Monitor.

Turns a conversation export (a file path or raw text, in JSON / markdown / plain
text) into a uniform ``list[dict]`` of ``{"role": ..., "text": ...}`` entries
where ``role`` is always ``"user"`` or ``"assistant"``.

The single public entry point is :func:`parse_transcript`. It is deliberately
forgiving: it NEVER raises on malformed input and returns a best-effort result
(possibly the empty list).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

__all__ = ["parse_transcript", "pick_anchor_goal"]

# Role synonym mapping. Anything not listed (and non-empty) falls back to "user".
_USER_ROLES = {"user", "human", "me", "you", "client", "customer", "prompt"}
_ASSISTANT_ROLES = {
    "assistant",
    "ai",
    "bot",
    "model",
    "gpt",
    "chatgpt",
    "claude",
    "system",
    "agent",
}

# Keys that may carry the role/text of a message inside a JSON object.
_ROLE_KEYS = ("role", "speaker", "author", "from", "sender", "name")
_TEXT_KEYS = ("text", "content", "message", "value", "body", "msg")

# Speaker marker at the start of a line. Two shapes are recognised:
#   1. label + separator + rest -> "User: hi", "AI - hello", "Assistant) ok"
#   2. bare label (no separator) -> "## User", "**Assistant**", "Human"
# In both cases group 1 is the speaker label and group 2 is any trailing text.
# A bare label is only accepted as a marker when it normalises to a known role
# (validated by `_is_marker_label`), so ordinary prose lines are not consumed.
_MARKER_RE = re.compile(
    r"^\s*(?:#{1,6}\s*|\*{1,2}\s*|>+\s*|-\s+)?"       # optional md heading/bullet/quote
    r"(?:"
    r"([A-Za-z][A-Za-z .]{0,30}?)\s*(?::|-|–|—|\))\s*(.*)"   # label + separator + rest
    r"|"
    r"([A-Za-z][A-Za-z ]{0,30}?)\s*\*{0,2}\s*"        # bare label (optional **bold**)
    r")$"
)

# Maximum reasonable file size to slurp when source looks like a path (8 MB).
_MAX_FILE_BYTES = 8 * 1024 * 1024


def _normalise_role(raw: Any, default: str = "user") -> str:
    """Map an arbitrary role/speaker token to ``"user"`` or ``"assistant"``."""
    if not isinstance(raw, str):
        return default
    token = raw.strip().lower()
    if not token:
        return default
    if token in _ASSISTANT_ROLES:
        return "assistant"
    if token in _USER_ROLES:
        return "user"
    # Substring heuristics for compound labels ("AI Assistant", "User 1", ...).
    if any(word in token for word in ("assist", "bot", "model", "agent")):
        return "assistant"
    if any(word in token for word in ("user", "human", "client", "customer")):
        return "user"
    return default


def _is_marker_label(label: str) -> str | None:
    """Return a normalised role if *label* is a recognised speaker marker, else None."""
    token = label.strip().lower()
    if token in _ASSISTANT_ROLES:
        return "assistant"
    if token in _USER_ROLES:
        return "user"
    return None


def _resolve_source(source: Any) -> str:
    """Return the text body to parse.

    If *source* is a string that names an existing file, read it; otherwise treat
    *source* as raw text. Any error degrades to an empty string.
    """
    if not isinstance(source, str):
        if source is None:
            return ""
        try:
            return str(source)
        except Exception:
            return ""
    # Detect a file path. Guard against absurdly long "paths" (which are really
    # raw transcript text containing newlines) before touching the filesystem.
    try:
        if "\n" not in source and len(source) < 4096 and os.path.isfile(source):
            if os.path.getsize(source) > _MAX_FILE_BYTES:
                return ""
            with open(source, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
    except (OSError, ValueError):
        pass
    return source


def _coerce_text(value: Any) -> str:
    """Best-effort conversion of a message payload to a single string."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in _TEXT_KEYS:
                    sub = item.get(key)
                    if isinstance(sub, str):
                        parts.append(sub)
                        break
        return "\n".join(p for p in parts if p).strip()
    if value is None:
        return ""
    try:
        return str(value).strip()
    except Exception:
        return ""


def _entry_from_obj(obj: dict) -> dict | None:
    """Build a ``{"role", "text"}`` entry from a JSON message object, or None."""
    role_raw: Any = None
    for key in _ROLE_KEYS:
        if key in obj and obj[key] is not None:
            role_raw = obj[key]
            break
    text_raw: Any = None
    for key in _TEXT_KEYS:
        if key in obj and obj[key] is not None:
            text_raw = obj[key]
            break
    text = _coerce_text(text_raw)
    if not text:
        return None
    return {"role": _normalise_role(role_raw), "text": text}


def _parse_json(body: str) -> list[dict] | None:
    """Try to parse *body* as a JSON transcript. Return None if it is not JSON."""
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return None

    items: Any
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = None
        for key in ("messages", "conversation", "turns", "history", "chat", "log"):
            value = data.get(key)
            if isinstance(value, list):
                items = value
                break
        if items is None:
            # A single message object, or nothing useful.
            entry = _entry_from_obj(data)
            return [entry] if entry else []
    else:
        return []

    result: list[dict] = []
    for item in items:
        if isinstance(item, dict):
            entry = _entry_from_obj(item)
            if entry:
                result.append(entry)
        elif isinstance(item, str):
            text = item.strip()
            if text:
                # Bare strings alternate user/assistant starting with user.
                role = "user" if len(result) % 2 == 0 else "assistant"
                result.append({"role": role, "text": text})
    return result


def _parse_markers(body: str) -> list[dict]:
    """Parse speaker-marker text (``User:`` / ``## Assistant`` / ``AI -`` ...).

    Returns ``[]`` when no recognised marker is present so the caller can fall
    back to alternation.
    """
    entries: list[dict] = []
    current_role: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_role is None:
            return
        text = "\n".join(buffer).strip()
        if text:
            entries.append({"role": current_role, "text": text})

    for line in body.splitlines():
        match = _MARKER_RE.match(line)
        role = None
        rest = ""
        if match:
            if match.group(1) is not None:
                # label + separator + rest
                role = _is_marker_label(match.group(1))
                rest = (match.group(2) or "").strip()
            elif match.group(3) is not None:
                # bare label (e.g. "## User"); accepted only if a known role.
                role = _is_marker_label(match.group(3))
                rest = ""
        if role is not None:
            # New speaker turn begins.
            flush()
            current_role = role
            buffer = []
            if rest:
                buffer.append(rest)
        else:
            if current_role is None:
                # Pre-amble before any marker is ignored for marker parsing.
                continue
            buffer.append(line)
    flush()
    return entries


def _parse_alternating(body: str) -> list[dict]:
    """Treat each non-empty line as an alternating user/assistant turn."""
    result: list[dict] = []
    for line in body.splitlines():
        text = line.strip()
        if not text:
            continue
        role = "user" if len(result) % 2 == 0 else "assistant"
        result.append({"role": role, "text": text})
    return result


def parse_transcript(source: str, fmt: str = "auto") -> list[dict]:
    """Parse a conversation export into ``[{"role", "text"}, ...]``.

    Args:
        source: Raw transcript text OR a path to a file. If the string names an
            existing readable file it is read; otherwise it is treated as text.
        fmt: One of ``"auto"`` | ``"json"`` | ``"markdown"`` | ``"text"``.
            ``"markdown"`` and ``"text"`` are handled identically (speaker
            markers with multi-line accumulation, falling back to alternation).
            ``"auto"`` tries JSON first, then marker parsing, then alternation.

    Returns:
        A list of ``{"role": "user"|"assistant", "text": str}`` dicts with
        whitespace trimmed and empty turns dropped. Never raises; returns a
        best-effort result (possibly ``[]``).
    """
    try:
        body = _resolve_source(source)
        if not body or not body.strip():
            return []

        fmt_norm = (fmt or "auto").strip().lower()

        if fmt_norm == "json":
            parsed = _parse_json(body)
            return parsed if parsed is not None else []

        if fmt_norm in ("markdown", "md", "text", "txt"):
            markers = _parse_markers(body)
            return markers if markers else _parse_alternating(body)

        # "auto" (and any unknown value): try JSON, then markers, then alternate.
        parsed = _parse_json(body)
        if parsed is not None:
            # Valid JSON. Trust it even if empty (an empty list is a valid
            # transcript) UNLESS it produced nothing AND the text clearly has
            # speaker markers we could use instead.
            if parsed:
                return parsed
            markers = _parse_markers(body)
            return markers if markers else parsed

        markers = _parse_markers(body)
        if markers:
            return markers
        return _parse_alternating(body)
    except Exception:
        # Absolute backstop: the contract forbids raising on any input.
        return []


# --- anchor selection --------------------------------------------------------
# A transcript's first line is often NOT its goal — on Claude Code / agent logs
# it is frequently a throwaway command ("delete this branch"), a branch name, a
# slash command, or a markup/log line. These patterns let pick_anchor_goal skip
# such turns and lock onto the first turn that reads like a stated objective.
_ANCHOR_BRANCH = re.compile(r"^[\w.\-]+/[\w.\-]+$")          # "cursor/feat-50f6"
_ANCHOR_SLASH_CMD = re.compile(r"^/[a-zA-Z][\w-]*")          # "/clear", "/compact"
_ANCHOR_SHELL_VERB = re.compile(
    r"^\s*(?:git|gh|npm|npx|pip|cd|ls|rm|mv|cp|cat|sudo|run|delete|remove|"
    r"merge|rebase|checkout|commit|push|pull|clone|stash|kill|chmod)\b",
    re.IGNORECASE,
)
_ANCHOR_MARKUP = re.compile(r"</?[a-zA-Z][\w-]*[>\s]")       # "<system-reminder>"
_ANCHOR_DELETE_BRANCH = re.compile(r"\bdelete\s+(?:this\s+)?branch\b", re.IGNORECASE)
_ANCHOR_WORD = re.compile(r"[A-Za-z]{2,}")


def _looks_like_goal(text: str) -> bool:
    """Heuristic: does *text* read like a stated objective rather than a
    throwaway command, branch name, slash command, or log/markup line?"""
    t = (text or "").strip()
    if not t:
        return False
    first_line = t.splitlines()[0].strip()
    if _ANCHOR_MARKUP.search(t):
        return False
    if _ANCHOR_SLASH_CMD.match(first_line):
        return False
    if _ANCHOR_BRANCH.match(first_line):
        return False
    if _ANCHOR_DELETE_BRANCH.search(t):
        return False
    words = _ANCHOR_WORD.findall(t)
    # A short shell-command-shaped turn is not a goal; longer prose may be.
    if _ANCHOR_SHELL_VERB.match(first_line) and len(words) < 12:
        return False
    # Require a little substance — a real goal is more than a word or two.
    return len(words) >= 4


def pick_anchor_goal(turns: list[dict], max_scan: int = 10, fallback: str = "") -> str:
    """Choose a sensible anchor goal from parsed transcript *turns*.

    The old behaviour — blindly taking ``turns[0]["text"]`` — anchors a session
    on whatever the first line happened to be, which on Claude Code / agent
    transcripts is often a throwaway command, a branch name, or a slash command
    (the literal-turn-1 mis-anchor bug). This walks the early **user** turns and
    returns the first that reads like an actual objective, falling back to the
    most substantial early turn, then the first turn, then *fallback*.

    Never raises; returns a trimmed string.
    """
    try:
        if not turns:
            return fallback
        users = [t for t in turns if str(t.get("role", "")).lower() == "user"]
        scan = (users or turns)[: max(1, max_scan)]
        for t in scan:
            text = str(t.get("text", "")).strip()
            if _looks_like_goal(text):
                return text
        # Nothing obviously goal-like: take the most substantial early turn.
        best = max(
            scan,
            key=lambda t: len(_ANCHOR_WORD.findall(str(t.get("text", "")))),
            default=None,
        )
        if best is not None:
            text = str(best.get("text", "")).strip()
            if text:
                return text
        return str(turns[0].get("text", "")).strip() or fallback
    except Exception:
        return fallback
