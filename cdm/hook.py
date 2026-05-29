"""Claude Code ``UserPromptSubmit`` hook — auto re-anchor when your terminal drifts.

Claude Code can run a hook each time you submit a prompt. This hook reads the hook
payload from stdin (it includes ``transcript_path``, ``cwd`` and the new ``prompt``),
measures how far the prompt has drifted from the session's original goal, and — only
when drift is genuinely high — prints a short re-anchor to stdout, which Claude Code
injects as extra context for that turn. When you're on track it prints nothing.

It uses the fast, offline hashing embedder so it adds negligible latency and never
needs the network (it runs in a fresh process on every prompt). It never blocks your
prompt: any error exits 0 silently.

Wire it up in ``~/.claude/settings.json``::

    {
      "hooks": {
        "UserPromptSubmit": [
          { "hooks": [ { "type": "command", "command": "drifter hook" } ] }
        ]
      }
    }
"""

from __future__ import annotations

import json
import sys
from typing import Dict, Optional

from cdm import claude_code as cc
from cdm.embeddings import get_embedder
from cdm.monitor import DriftMonitor
from cdm.storage import Store


def run_hook(payload: Dict, threshold: Optional[float] = None, embedder=None) -> Dict:
    """Evaluate drift for a UserPromptSubmit payload.

    Args:
        payload: The Claude Code hook payload (``prompt``, ``transcript_path``).
        threshold: Optional absolute drift threshold (mostly vestigial now that the
            verdict is self-calibrating; retained for API compatibility).
        embedder: Embedding backend to score with — an :class:`~cdm.embeddings.Embedder`
            instance, an embedder name (e.g. ``"hashing"`` / ``"semantic"``), or
            ``None`` for the default fast offline hashing embedder. The hashing
            embedder can't separate on- from off-goal turns, so the hook stays
            silent on it; pass ``"semantic"`` for an actually-useful hook.

    Returns ``{"high": bool, "drift": float, "context": str | None}``. ``context``
    is the re-anchor text to inject (only when drift is high).
    """
    prompt = (payload.get("prompt") or "").strip()
    transcript = payload.get("transcript_path") or ""
    if not prompt:
        return {"high": False, "drift": 0.0, "context": None}

    turns = cc.parse_transcript_file(transcript) if transcript else []
    user_turns = [t for t in turns if t["role"] == "user"]
    if not user_turns:
        # No prior goal to anchor against yet (this is likely the first prompt).
        return {"high": False, "drift": 0.0, "context": None}

    anchor = user_turns[0]["text"]
    if embedder is None:
        embedder = get_embedder("hashing")  # fast + offline; the hook runs per-prompt
    elif isinstance(embedder, str):
        embedder = get_embedder(embedder)
    thr = threshold if threshold is not None else getattr(embedder, "suggested_threshold", 0.8)
    monitor = DriftMonitor(store=Store(":memory:"), embedder=embedder, threshold=thr)
    session = monitor.start_session("hook", anchor, [])
    rest = [{"role": t["role"], "text": t["text"]} for t in turns[1:]]
    if rest:
        monitor.ingest_transcript(session.session_id, rest)
    monitor.add_turn(session.session_id, "user", prompt)
    # Use the self-calibrating verdict, not the raw per-turn threshold: the latter
    # fires on any lexically-divergent on-goal prompt (and the hook runs on the
    # fast hashing embedder, where on/off-goal turns share a distance band), so an
    # absolute cutoff would re-anchor on nearly every prompt. This flags only a
    # sustained departure from the conversation's own baseline.
    verdict = monitor.is_drifting_calibrated(session.session_id)
    high = bool(verdict["high"])
    context = monitor.current_corrective_prompt(session.session_id) if high else None
    return {"high": high, "drift": float(verdict["drift"]), "context": context}


def main(argv: Optional[list] = None) -> int:
    """stdin → drift check → optional re-anchor on stdout. Always exits 0."""
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        out = run_hook(payload)
        if out.get("context"):
            sys.stdout.write(
                "[Drifter] Heads up — this turn looks off-track from the original goal. "
                "Consider this re-anchor:\n\n" + out["context"] + "\n"
            )
    except Exception:
        # Never block the user's prompt because of the monitor.
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
