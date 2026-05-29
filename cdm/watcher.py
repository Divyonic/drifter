"""Clipboard auto-capture for the Context Drift Monitor.

The watcher runs as its own background process. It polls the system clipboard and,
whenever you copy a new chunk of text (a prompt you send or a reply you receive),
appends it as a turn to the currently *active* session — so the live drift graph
fills in by itself while you chat with any LLM (claude.ai, ChatGPT, a local model,
…). It is fully offline: clipboard access is local, scoring is local.

Two layers:

* :class:`ClipboardWatcher` — the testable core. ``poll_once(text)`` takes the
  current clipboard contents and decides whether to ingest it. No real clipboard or
  sleeping is involved, so it is unit-testable with plain strings.
* Process helpers (:func:`start_watcher_process`, :func:`stop_watcher`,
  :func:`is_watcher_running`, :func:`watcher_heartbeat_age`) — let the Streamlit app
  spawn, monitor and stop the watcher across reruns via a pidfile + a DB heartbeat.

Role assignment alternates based on the *stored* history (so it is robust to
restarts): the next captured turn takes the opposite role of the last stored turn,
defaulting to ``start_role`` for the very first turn. Drift-vs-anchor scoring — the
headline signal — does not depend on the role being perfect.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from cdm import config
from cdm.corrective import CORRECTIVE_TEMPLATE
from cdm.monitor import DriftMonitor
from cdm.storage import Store

__all__ = [
    "ClipboardWatcher",
    "read_clipboard",
    "start_watcher_process",
    "stop_watcher",
    "is_watcher_running",
    "watcher_heartbeat_age",
    "main",
]

# First line of the corrective prompt — used to ignore it if the user copies it
# (otherwise pasting the corrective prompt would itself be captured as a turn).
_CORRECTIVE_PREFIX = CORRECTIVE_TEMPLATE.split("\n", 1)[0].strip()

# Meta keys (shared via the Store across the app and watcher processes).
_K_STATUS = "watcher_status"
_K_HEARTBEAT = "watcher_heartbeat"


def read_clipboard() -> Optional[str]:
    """Return the current clipboard text, or ``None`` if it cannot be read.

    Uses :mod:`pyperclip` (which shells out to the platform's native clipboard:
    ``pbcopy``/``pbpaste`` on macOS, ``clip``/PowerShell on Windows,
    ``xclip``/``xsel`` on Linux). Any failure returns ``None`` rather than raising
    so the polling loop never dies.
    """
    try:
        import pyperclip  # lazy: keeps the package importable without it
    except Exception:
        return None
    try:
        value = pyperclip.paste()
    except Exception:
        return None
    return value if isinstance(value, str) else None


class ClipboardWatcher:
    """Capture clipboard changes into the active session as conversation turns."""

    def __init__(
        self,
        store: Optional[Store] = None,
        monitor: Optional[DriftMonitor] = None,
        poll_interval: float = 1.0,
        min_chars: int = 10,
        min_words: int = 2,
        max_chars: int = 20000,
        start_role: str = "user",
    ) -> None:
        """Build a watcher.

        Args:
            store: Shared :class:`Store`. Defaults to one at the configured DB path.
            monitor: :class:`DriftMonitor` to score with. Defaults to one sharing
                ``store`` (and the auto-selected offline embedder).
            poll_interval: Seconds between clipboard polls in :meth:`run`.
            min_chars: Ignore clipboard text shorter than this (after trimming).
            min_words: Ignore clipboard text with fewer than this many words.
            max_chars: Truncate captured text to this length (guards giant copies).
            start_role: Role assigned to the very first captured turn of a session.
        """
        self.store: Store = store if store is not None else Store()
        self.monitor: DriftMonitor = (
            monitor if monitor is not None else DriftMonitor(store=self.store)
        )
        self.poll_interval = float(poll_interval)
        self.min_chars = int(min_chars)
        self.min_words = int(min_words)
        self.max_chars = int(max_chars)
        self.start_role = start_role
        self._last_clip: Optional[str] = None

    # -- decision logic (pure / testable) ------------------------------------

    def _should_capture(self, text: str) -> bool:
        """Return True if ``text`` looks like a real conversation turn worth scoring."""
        stripped = text.strip()
        if len(stripped) < self.min_chars:
            return False
        if len(stripped.split()) < self.min_words:
            return False
        # Don't capture the corrective prompt itself (avoids a feedback loop).
        if stripped.startswith(_CORRECTIVE_PREFIX):
            return False
        return True

    def _next_role(self, session_id: str) -> str:
        """Alternate role from the last stored turn (robust across restarts)."""
        messages = self.store.get_messages(session_id)
        if not messages:
            return self.start_role
        return "assistant" if (messages[-1].role or "").lower() == "user" else "user"

    def capture(self, text: str) -> Optional[dict]:
        """Ingest one clipboard payload into the active session.

        Returns the :meth:`DriftMonitor.add_turn` result dict, or ``None`` when
        there is no active session or the text was filtered / a duplicate.
        """
        session_id = self.store.get_active_session()
        if not session_id or self.store.get_session(session_id) is None:
            return None
        if not self._should_capture(text):
            return None
        stripped = text.strip()[: self.max_chars]
        # Skip an exact repeat of the most recent stored turn (e.g. copied twice).
        messages = self.store.get_messages(session_id)
        if messages and (messages[-1].text or "").strip() == stripped:
            return None
        role = self._next_role(session_id)
        return self.monitor.add_turn(session_id, role, stripped)

    def poll_once(self, clip_text: Optional[str]) -> Optional[dict]:
        """Process the current clipboard value; capture it only if it changed."""
        if clip_text is None or clip_text == self._last_clip:
            return None
        self._last_clip = clip_text
        return self.capture(clip_text)

    # -- run loop ------------------------------------------------------------

    def _heartbeat(self, status: str) -> None:
        """Record liveness so the app can show a 'watching' indicator."""
        try:
            self.store.set_meta(_K_STATUS, status)
            self.store.set_meta(_K_HEARTBEAT, datetime.now().isoformat())
        except Exception:
            pass

    def run(self, stop_when=None) -> None:
        """Poll the clipboard until ``stop_when()`` returns True (or forever).

        Each iteration reads the clipboard, ingests it if new/valid, writes a
        heartbeat, then sleeps ``poll_interval``. A bad turn never stops the loop.
        """
        self._heartbeat("running")
        try:
            while not (stop_when and stop_when()):
                try:
                    self.poll_once(read_clipboard())
                except Exception:
                    pass  # never let one bad capture kill the watcher
                self._heartbeat("running")
                time.sleep(self.poll_interval)
        finally:
            self._heartbeat("stopped")


# --------------------------------------------------------------------------- #
# Process management (used by the Streamlit app)
# --------------------------------------------------------------------------- #
def _pidfile() -> Path:
    """Path to the watcher pidfile inside the data directory."""
    config.ensure_data_dir()
    return config.DATA_DIR / "watcher.pid"


def _pid_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` is currently alive."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_pid() -> Optional[int]:
    """Read the pid from the pidfile, or ``None`` if absent/invalid."""
    pf = _pidfile()
    if not pf.exists():
        return None
    try:
        return int(pf.read_text().strip())
    except Exception:
        return None


def is_watcher_running() -> bool:
    """True if a watcher process recorded in the pidfile is alive."""
    pid = _read_pid()
    return bool(pid and _pid_alive(pid))


def watcher_heartbeat_age(store: Optional[Store] = None) -> Optional[float]:
    """Seconds since the watcher's last heartbeat, or ``None`` if never seen."""
    store = store or Store()
    iso = store.get_meta(_K_HEARTBEAT)
    if not iso:
        return None
    try:
        beat = datetime.fromisoformat(iso)
    except Exception:
        return None
    return max(0.0, (datetime.now() - beat).total_seconds())


def start_watcher_process(db_path: Optional[str] = None) -> Optional[int]:
    """Spawn ``python -m cdm.watcher`` as a detached background process.

    Writes its pid to the pidfile immediately (so the app can detect/stop it
    without a race). Returns the pid, or the existing pid if one is already alive.
    """
    if is_watcher_running():
        return _read_pid()
    env = os.environ.copy()
    if db_path:
        env["CDM_DB_PATH"] = str(db_path)
    proc = subprocess.Popen(
        [sys.executable, "-m", "cdm.watcher"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # detach from the app so it survives reruns
    )
    try:
        _pidfile().write_text(str(proc.pid))
    except Exception:
        pass
    return proc.pid


def stop_watcher() -> bool:
    """Terminate the running watcher (if any) and remove the pidfile."""
    pid = _read_pid()
    stopped = False
    if pid and _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            stopped = True
        except OSError:
            pass
    try:
        _pidfile().unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass
    try:
        Store().set_meta(_K_STATUS, "stopped")
    except Exception:
        pass
    return stopped


def main(argv: Optional[list] = None) -> int:
    """Entry point for ``python -m cdm.watcher`` / ``drifter watch``.

    Runs the clipboard watch loop until interrupted (SIGTERM/SIGINT). Cleans up
    the pidfile and heartbeat on exit.
    """
    pf = _pidfile()
    try:
        pf.write_text(str(os.getpid()))
    except Exception:
        pass

    stop = {"flag": False}

    def _handle(signum, frame):  # noqa: ANN001 - signal handler signature
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    watcher = ClipboardWatcher()
    try:
        watcher.run(stop_when=lambda: stop["flag"])
    finally:
        try:
            if _read_pid() == os.getpid():
                pf.unlink(missing_ok=True)  # type: ignore[call-arg]
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
