"""Read & tail Claude Code session transcripts so Drifter can monitor a real
terminal conversation (you chat in `claude`, Drifter just draws the drift graph).

Claude Code stores each session as JSON-Lines at
``~/.claude/projects/<encoded-cwd>/<session-id>.jsonl``. Each line is an event; the
ones we care about are ``type: "user"`` (``message.content`` is the typed prompt
string) and ``type: "assistant"`` (``message.content`` is a list of blocks whose
``text`` blocks are the visible reply). Sub-agent sidechains and tool-only turns are
skipped. Everything here is local file reading of the user's own data.
"""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set

__all__ = [
    "projects_dir",
    "list_sessions",
    "extract_turn",
    "parse_transcript_file",
    "ClaudeCodeTail",
    "snapshot_transcripts",
    "find_new_transcript",
    "launch_claude_in_terminal",
]

# Noise we never want to treat as a real user turn (slash-command echoes, the
# local-command caveat wrapper, system reminders pasted as user text).
_SKIP_PREFIXES = ("<command-", "<local-command", "Caveat: The messages below")


def projects_dir() -> Path:
    """Directory holding Claude Code project transcripts."""
    base = os.environ.get("CLAUDE_CONFIG_DIR") or str(Path.home() / ".claude")
    return Path(base) / "projects"


def _text_from_message(msg: dict) -> str:
    """Pull visible text out of a message (string content, or 'text' blocks)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p).strip()
    return ""


def extract_turn(obj: dict) -> Optional[Dict[str, str]]:
    """Return ``{role, text, uuid, ts}`` for a transcript line, or None to skip.

    Skips sidechains, meta/system lines, tool-only turns and command-echo noise.
    """
    if not isinstance(obj, dict) or obj.get("isSidechain") or obj.get("isMeta"):
        return None
    if obj.get("type") not in ("user", "assistant"):
        return None
    msg = obj.get("message")
    if not isinstance(msg, dict) or msg.get("role") not in ("user", "assistant"):
        return None
    text = _text_from_message(msg)
    if not text or text.startswith(_SKIP_PREFIXES):
        return None
    return {
        "role": msg["role"],
        "text": text,
        "uuid": obj.get("uuid", ""),
        "ts": obj.get("timestamp", ""),
    }


def parse_transcript_file(path) -> List[Dict[str, str]]:
    """Parse a whole transcript file into ordered turns (deduped by uuid)."""
    turns: List[Dict[str, str]] = []
    seen = set()
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                turn = extract_turn(obj)
                if turn and turn["uuid"] not in seen:
                    seen.add(turn["uuid"])
                    turns.append(turn)
    except FileNotFoundError:
        return []
    return turns


def _first_user_prompt(path, max_lines: int = 200) -> str:
    """Cheap label: the first typed user prompt (scans a bounded prefix)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for _i, line in zip(range(max_lines), fh):
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                turn = extract_turn(obj)
                if turn and turn["role"] == "user":
                    return turn["text"]
    except FileNotFoundError:
        return ""
    return ""


def _cwd_of(path, max_lines: int = 40) -> str:
    """Read the working directory recorded in the transcript, if present."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for _i, line in zip(range(max_lines), fh):
                try:
                    cwd = json.loads(line).get("cwd")
                except Exception:
                    continue
                if cwd:
                    return cwd
    except FileNotFoundError:
        return ""
    return ""


def list_sessions(limit: int = 40) -> List[Dict]:
    """List Claude Code sessions across all projects, newest first.

    Each entry: ``{path, session_id, cwd, title, mtime}``. ``title`` is the first
    user prompt (truncated); ``cwd`` is read from the transcript when available.
    """
    root = projects_dir()
    if not root.exists():
        return []
    files = []
    for proj in root.iterdir():
        if not proj.is_dir():
            continue
        for jsonl in proj.glob("*.jsonl"):
            try:
                files.append((jsonl.stat().st_mtime, jsonl))
            except OSError:
                continue
    files.sort(key=lambda t: t[0], reverse=True)

    out: List[Dict] = []
    for mtime, path in files[:limit]:
        prompt = _first_user_prompt(path)
        if not prompt:
            continue  # skip empty/non-conversation transcripts
        cwd = _cwd_of(path)
        out.append({
            "path": str(path),
            "session_id": path.stem,
            "cwd": cwd or path.parent.name,
            "title": (prompt[:70] + "…") if len(prompt) > 70 else prompt,
            "mtime": mtime,
        })
    return out


def snapshot_transcripts() -> Set[str]:
    """Set of all transcript paths under the projects dir (call before launching)."""
    root = projects_dir()
    if not root.exists():
        return set()
    out: Set[str] = set()
    for proj in root.iterdir():
        if proj.is_dir():
            out.update(str(p) for p in proj.glob("*.jsonl"))
    return out


def find_new_transcript(before: Set[str], cwd: Optional[str] = None) -> Optional[str]:
    """Newest transcript that wasn't in ``before`` (prefer one whose cwd matches)."""
    root = projects_dir()
    if not root.exists():
        return None
    fresh = []
    for proj in root.iterdir():
        if not proj.is_dir():
            continue
        for p in proj.glob("*.jsonl"):
            sp = str(p)
            if sp in before:
                continue
            try:
                fresh.append((p.stat().st_mtime, sp))
            except OSError:
                continue
    if not fresh:
        return None
    fresh.sort(reverse=True)
    if cwd:
        for _mt, sp in fresh:
            if _cwd_of(sp) == cwd:
                return sp
    return fresh[0][1]


def launch_claude_in_terminal(
    cwd: str, kickoff: str, anchor: Optional[str] = None, name: Optional[str] = None
) -> bool:
    """Open a new Terminal window running an interactive ``claude`` session.

    Seeds it with ``kickoff`` (first prompt), ``anchor`` (appended system prompt) and
    an optional session ``name``. macOS only (uses Terminal.app via osascript);
    returns False on other platforms or any failure so the caller can degrade.
    """
    if platform.system() != "Darwin":
        return False
    exe = shutil.which("claude") or "claude"
    args = [exe]
    if name:
        args += ["-n", name]
    if anchor:
        args += ["--append-system-prompt", anchor]
    if kickoff:
        args += [kickoff]
    command = "cd " + shlex.quote(cwd) + " && " + " ".join(shlex.quote(a) for a in args)
    escaped = command.replace("\\", "\\\\").replace('"', '\\"')
    script = f'tell application "Terminal"\n  activate\n  do script "{escaped}"\nend tell'
    try:
        subprocess.run(["osascript", "-e", script], check=True, capture_output=True, timeout=15)
        return True
    except Exception:
        return False


class ClaudeCodeTail:
    """Incrementally yield new turns appended to a transcript file.

    ``start_at_end=True`` begins after the current end of file (so only turns
    written *after* monitoring starts are returned). Tracks a byte offset and only
    consumes complete (newline-terminated) lines, so a half-written final line is
    re-read on the next poll.
    """

    def __init__(self, path, start_at_end: bool = False) -> None:
        self.path = str(path)
        self._seen = set()
        self._pos = 0
        if start_at_end:
            try:
                self._pos = os.path.getsize(self.path)
            except OSError:
                self._pos = 0

    def new_turns(self) -> List[Dict[str, str]]:
        """Return turns appended since the last call (possibly empty)."""
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self._pos)
                chunk = fh.read()
        except FileNotFoundError:
            return []
        if b"\n" not in chunk:
            return []
        complete, _, _partial = chunk.rpartition(b"\n")
        self._pos += len(complete) + 1  # advance past consumed lines + newline
        turns: List[Dict[str, str]] = []
        for raw in complete.split(b"\n"):
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            turn = extract_turn(obj)
            if turn and turn["uuid"] not in self._seen:
                self._seen.add(turn["uuid"])
                turns.append(turn)
        return turns
