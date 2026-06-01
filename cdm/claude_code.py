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
    "send_to_terminal",
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


# --- launching & forwarding into an interactive `claude` session -------------
#
# Drifter can open a `claude` session in a real terminal and then forward messages
# the user types in the app INTO that session. This is cross-platform via a small
# backend dispatch keyed by a scheme-prefixed *handle* string:
#
#   "tmux:<session>"        -> tmux send-keys (macOS / Linux / WSL / MSYS — preferred)
#   "applescript:<tty>"     -> Terminal.app via osascript (macOS fallback, no tmux)
#
# tmux is preferred everywhere it exists: `send-keys` is robust, needs no
# Accessibility/Automation permission, and takes text as a literal argv argument
# (no escaping pitfalls). The reply always flows back into Drifter via the
# transcript tail regardless of backend.

# Linux terminal emulators and the flag each uses to run a command, tried in order.
_LINUX_TERMINALS: List[tuple] = [
    ("x-terminal-emulator", ["-e"]),
    ("gnome-terminal", ["--"]),
    ("konsole", ["-e"]),
    ("xfce4-terminal", ["-x"]),
    ("tilix", ["-e"]),
    ("alacritty", ["-e"]),
    ("kitty", []),
    ("xterm", ["-e"]),
]


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _claude_argv(name: Optional[str], anchor: Optional[str], kickoff: Optional[str]) -> List[str]:
    """The argv to start an interactive ``claude`` seeded with goal/kickoff."""
    argv = [shutil.which("claude") or "claude"]
    if name:
        argv += ["-n", name]
    if anchor:
        argv += ["--append-system-prompt", anchor]
    if kickoff:
        argv += [kickoff]
    return argv


def _tmux_session_name(name: Optional[str]) -> str:
    import re
    import uuid

    base = re.sub(r"[^A-Za-z0-9_]", "_", (name or "drifter")).strip("_")[:24] or "drifter"
    return f"drifter_{base}_{uuid.uuid4().hex[:6]}"


def _open_terminal_attached(session: str) -> bool:
    """Best-effort: open a visible terminal attached to the tmux ``session``.

    Forwarding works without this (the user can ``tmux attach`` themselves and
    Drifter tails the transcript regardless), so failures are non-fatal.
    """
    attach = ["tmux", "attach", "-t", session]
    sysname = platform.system()
    try:
        if sysname == "Darwin":
            cmd = "tmux attach -t " + shlex.quote(session)
            esc = cmd.replace("\\", "\\\\").replace('"', '\\"')
            subprocess.run(
                ["osascript", "-e",
                 f'tell application "Terminal"\n  activate\n  do script "{esc}"\nend tell'],
                capture_output=True, timeout=15,
            )
            return True
        if sysname == "Linux":
            for term, flags in _LINUX_TERMINALS:
                if _have(term):
                    try:
                        subprocess.Popen([term, *flags, *attach])
                        return True
                    except Exception:
                        continue
            return False
        if sysname == "Windows":
            # Prefer Windows Terminal; fall back to a new console window.
            for prefix in (["wt", "--"], ["cmd", "/c", "start", ""]):
                if prefix[0] == "wt" and not _have("wt"):
                    continue
                try:
                    subprocess.Popen([*prefix, *attach])
                    return True
                except Exception:
                    continue
            return False
    except Exception:
        return False
    return False


def _launch_tmux(cwd: str, name: Optional[str], anchor: Optional[str],
                 kickoff: Optional[str]) -> Optional[str]:
    """Start ``claude`` in a detached tmux session; return its ``tmux:<name>`` handle."""
    session = _tmux_session_name(name)
    argv = _claude_argv(name, anchor, kickoff)
    # tmux runs this via the shell, so quote for the shell and exec into claude.
    cmd = "cd " + shlex.quote(cwd) + " && exec " + " ".join(shlex.quote(a) for a in argv)
    try:
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, cmd],
            check=True, capture_output=True, timeout=15,
        )
    except Exception:
        return None
    _open_terminal_attached(session)  # best-effort visibility
    return f"tmux:{session}"


def _launch_applescript(cwd: str, name: Optional[str], anchor: Optional[str],
                        kickoff: Optional[str]) -> tuple[bool, Optional[str]]:
    """macOS fallback: open Terminal.app via osascript and capture the tab's tty."""
    argv = _claude_argv(name, anchor, kickoff)
    command = "cd " + shlex.quote(cwd) + " && " + " ".join(shlex.quote(a) for a in argv)
    escaped = command.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        'tell application "Terminal"\n'
        "  activate\n"
        f'  set _t to do script "{escaped}"\n'
        "  return tty of _t\n"
        "end tell"
    )
    try:
        r = subprocess.run(
            ["osascript", "-e", script], check=True, capture_output=True, timeout=15, text=True
        )
        tty = (r.stdout or "").strip()
        # Launched either way; only a real tty enables forwarding.
        return (True, f"applescript:{tty}" if tty.startswith("/dev/") else None)
    except Exception:
        return (False, None)


def launch_claude_in_terminal(
    cwd: str, kickoff: str, anchor: Optional[str] = None, name: Optional[str] = None
) -> tuple[bool, Optional[str]]:
    """Open an interactive ``claude`` session in a terminal, cross-platform.

    Seeds it with ``kickoff`` (first prompt), ``anchor`` (appended system prompt) and
    an optional session ``name``. Prefers tmux on any platform (so messages can be
    forwarded later via :func:`send_to_terminal`), with a macOS Terminal.app fallback.

    Returns ``(launched, handle)``: ``launched`` is True on success; ``handle`` is a
    scheme-prefixed string (``"tmux:..."`` / ``"applescript:..."``) the caller passes
    back to :func:`send_to_terminal`, or None when forwarding isn't available.
    """
    if _have("tmux"):
        handle = _launch_tmux(cwd, name, anchor, kickoff)
        if handle:
            return (True, handle)
    if platform.system() == "Darwin":
        return _launch_applescript(cwd, name, anchor, kickoff)
    return (False, None)


def _send_tmux(session: str, line: str) -> bool:
    try:
        subprocess.run(["tmux", "send-keys", "-t", session, "-l", line],
                       check=True, capture_output=True, timeout=10)
        subprocess.run(["tmux", "send-keys", "-t", session, "Enter"],
                       check=True, capture_output=True, timeout=10)
        return True
    except Exception:
        return False


def _applescript_send_script(tty: str, line: str) -> str:
    """AppleScript that types ``line`` into the Terminal tab on ``tty`` (then submits)."""
    esc = line.replace("\\", "\\\\").replace('"', '\\"')
    tty_esc = tty.replace("\\", "\\\\").replace('"', '\\"')
    return (
        'tell application "Terminal"\n'
        "  repeat with w in windows\n"
        "    repeat with t in tabs of w\n"
        f'      if tty of t is "{tty_esc}" then\n'
        f'        do script "{esc}" in t\n'
        '        return "ok"\n'
        "      end if\n"
        "    end repeat\n"
        "  end repeat\n"
        "end tell\n"
        'return "no"'
    )


def _send_applescript(tty: str, line: str) -> bool:
    if platform.system() != "Darwin" or not tty:
        return False
    try:
        r = subprocess.run(
            ["osascript", "-e", _applescript_send_script(tty, line)],
            capture_output=True, timeout=10, text=True,
        )
        return r.returncode == 0 and "ok" in (r.stdout or "")
    except Exception:
        return False


def send_to_terminal(handle: str, text: str) -> bool:
    """Forward ``text`` as a prompt into the linked ``claude`` session.

    ``handle`` is what :func:`launch_claude_in_terminal` returned. Newlines are
    collapsed to a single line because a TUI submits on Enter (an embedded newline
    would send early). Returns True when delivered; the reply tails back into Drifter.
    """
    if not handle or not text.strip():
        return False
    line = " ".join(text.split())
    scheme, _, rest = handle.partition(":")
    if scheme == "tmux":
        return _send_tmux(rest, line)
    if scheme == "applescript":
        return _send_applescript(rest, line)
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
