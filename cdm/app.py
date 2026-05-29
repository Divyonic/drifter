"""Streamlit UI for the Context Drift Monitor ("Drifter").

A single-file app over :class:`cdm.monitor.DriftMonitor`. It can capture your
conversation **automatically**: a background clipboard watcher (see
:mod:`cdm.watcher`) appends every prompt/reply you copy to the active session, and
the drift graph here **refreshes itself live** (via an auto-rerunning fragment) so
you just chat with your LLM and watch drift climb in real time. Sessions persist
and auto-resume across launches. You can also feed turns manually.

Run with ``drifter`` (installed), ``streamlit run app.py`` (repo root shim), or
``streamlit run cdm/app.py``. Fully offline — no API keys.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from cdm import config
from cdm.embeddings import Embedder, get_embedder
from cdm.monitor import DriftMonitor
from cdm.transcript import parse_transcript
from cdm.watcher import (
    is_watcher_running,
    start_watcher_process,
    stop_watcher,
    watcher_heartbeat_age,
)

# Threshold slider bounds (the contract specifies 0.30-0.95).
_THRESHOLD_MIN = 0.30
_THRESHOLD_MAX = 0.95
_THRESHOLD_STEP = 0.01

# Live-graph auto-refresh cadence (seconds).
_REFRESH_SECONDS = 2.0
# A watcher whose last heartbeat is within this many seconds is "live".
_HEARTBEAT_FRESH = 6.0


def _clipboard_available() -> bool:
    """True if pyperclip (the clipboard backend) is importable."""
    try:
        import pyperclip  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _build_embedder() -> Embedder:
    """Build (once per process) the embedding backend."""
    return get_embedder()


@st.cache_resource(show_spinner=False)
def _get_monitor() -> DriftMonitor:
    """Return the shared :class:`DriftMonitor`, reusing the cached embedder."""
    return DriftMonitor(embedder=_build_embedder())


def _is_hashing(embedder: Embedder) -> bool:
    """Return True when the active embedder is the pure-Python hashing fallback."""
    return str(getattr(embedder, "name", "")).lower().startswith("hashing")


def _demo_transcript_path() -> Optional[str]:
    """Path to the bundled demo transcript, or None if it isn't packaged."""
    from pathlib import Path

    candidate = Path(__file__).resolve().parent / "sample_transcript.json"
    return str(candidate) if candidate.exists() else None


def _load_demo(monitor: DriftMonitor) -> Optional[str]:
    """Create a demo session from the bundled transcript; return its id."""
    path = _demo_transcript_path()
    if not path:
        return None
    turns = parse_transcript(path)
    if not turns:
        return None
    session = monitor.start_session(
        "Demo: pan-tilt gimbal",
        turns[0]["text"],
        ["mass < 5 kg", "volume < 10 L", "COTS stepper motors"],
    )
    if len(turns) > 1:
        monitor.ingest_transcript(session.session_id, turns[1:])
    return session.session_id


# --------------------------------------------------------------------------- #
# Session-state helpers (active session is mirrored into the DB for the watcher)
# --------------------------------------------------------------------------- #
def _active_session_id() -> Optional[str]:
    """Return the active session id from session state."""
    return st.session_state.get("active_session_id")


def _set_active_session(monitor: DriftMonitor, session_id: Optional[str]) -> None:
    """Set the active session in state AND in the DB so the watcher follows it."""
    st.session_state["active_session_id"] = session_id
    try:
        monitor.store.set_active_session(session_id)
    except Exception:  # pragma: no cover - defensive
        pass


def _resume_active_session(monitor: DriftMonitor) -> None:
    """On first load, resume the session the watcher/last run left active."""
    if "active_session_id" in st.session_state:
        return
    try:
        sid = monitor.store.get_active_session()
        if sid and monitor.store.get_session(sid) is not None:
            st.session_state["active_session_id"] = sid
        else:
            st.session_state["active_session_id"] = None
    except Exception:  # pragma: no cover - defensive
        st.session_state["active_session_id"] = None


def _safe_list_sessions(monitor: DriftMonitor):
    """List persisted sessions, returning [] on any storage hiccup."""
    try:
        return monitor.store.list_sessions()
    except Exception:  # pragma: no cover - defensive UI guard
        return []


# --------------------------------------------------------------------------- #
# Watcher control
# --------------------------------------------------------------------------- #
def _watcher_is_live(monitor: DriftMonitor) -> bool:
    """True if the watcher process is alive AND heartbeating recently."""
    if not is_watcher_running():
        return False
    age = watcher_heartbeat_age(monitor.store)
    return age is not None and age <= _HEARTBEAT_FRESH


def _ensure_watcher(monitor: DriftMonitor) -> None:
    """Start the clipboard watcher if it isn't already running."""
    if not is_watcher_running():
        start_watcher_process(db_path=monitor.store.db_path)


def _render_watcher_controls(monitor: DriftMonitor) -> None:
    """Sidebar block: clipboard auto-capture status + start/stop + auto toggle."""
    st.subheader("Auto-capture")

    if not _clipboard_available():
        st.warning(
            "Clipboard backend not installed. Run `pip install pyperclip` to "
            "enable automatic capture (manual entry still works)."
        )

    auto = st.checkbox(
        "Capture clipboard automatically",
        value=st.session_state.get("auto_capture", True),
        help="When on, copying any prompt or reply (Cmd/Ctrl+C) adds it as a turn.",
    )
    st.session_state["auto_capture"] = auto

    # Auto-start the watcher when enabled and a session is active.
    if auto and _active_session_id() and _clipboard_available():
        _ensure_watcher(monitor)

    live = _watcher_is_live(monitor)
    if live:
        st.success("● Watching your clipboard — copy a message to log it.")
    elif is_watcher_running():
        st.info("Watcher starting…")
    else:
        st.caption("Watcher idle.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start", key="start_watch", disabled=not _clipboard_available()):
            _ensure_watcher(monitor)
            st.rerun()
    with col2:
        if st.button("Stop", key="stop_watch"):
            stop_watcher()
            st.rerun()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _render_sidebar(monitor: DriftMonitor, embedder: Embedder) -> None:
    """Render the sidebar: new-session form, selector, watcher, threshold, info."""
    with st.sidebar:
        st.header("Context Drift Monitor")

        # -- New session form ------------------------------------------------ #
        with st.form("new_session_form", clear_on_submit=True):
            st.subheader("New session")
            project_name = st.text_input("Project name", value="")
            initial_goal = st.text_area(
                "Initial goal (anchor)",
                value="",
                height=100,
                help="The north-star goal the conversation should stay on.",
            )
            constraints_raw = st.text_area(
                "Constraints (one per line)",
                value="",
                height=100,
                help="Optional hard requirements, e.g. 'under 5 kg'.",
            )
            submitted = st.form_submit_button("Start session")

        if submitted:
            goal = initial_goal.strip()
            name = project_name.strip() or "Untitled project"
            if not goal:
                st.error("An initial goal is required to start a session.")
            else:
                constraints = [
                    line.strip()
                    for line in constraints_raw.splitlines()
                    if line.strip()
                ]
                try:
                    session = monitor.start_session(name, goal, constraints)
                    _set_active_session(monitor, session.session_id)
                    if st.session_state.get("auto_capture", True) and _clipboard_available():
                        _ensure_watcher(monitor)
                    st.success(f"Started session: {name}")
                    st.rerun()
                except Exception as exc:  # pragma: no cover - defensive UI guard
                    st.error(f"Could not start session: {exc}")

        # -- One-click demo -------------------------------------------------- #
        if _demo_transcript_path() and st.button(
            "Load demo session", help="See drift in action on a sample conversation."
        ):
            try:
                sid = _load_demo(monitor)
                if sid:
                    _set_active_session(monitor, sid)
                    st.rerun()
                else:
                    st.error("Demo transcript could not be loaded.")
            except Exception as exc:  # pragma: no cover - defensive UI guard
                st.error(f"Could not load demo: {exc}")

        # -- Existing-session selector -------------------------------------- #
        st.subheader("Sessions")
        sessions = _safe_list_sessions(monitor)
        if sessions:
            labels = {
                s.session_id: f"{s.project_name}  ·  {s.session_id[:8]}"
                for s in sessions
            }
            ids = [s.session_id for s in sessions]
            active = _active_session_id()
            index = ids.index(active) if active in ids else 0
            chosen = st.selectbox(
                "Continue a session",
                options=ids,
                index=index,
                format_func=lambda sid: labels.get(sid, sid),
            )
            if chosen != active:
                _set_active_session(monitor, chosen)
                st.rerun()
        else:
            st.caption("No sessions yet — create one above.")

        st.divider()
        _render_watcher_controls(monitor)

        # -- Threshold slider ------------------------------------------------ #
        st.divider()
        st.subheader("Drift threshold")
        default_threshold = float(
            st.session_state.get("threshold", monitor.threshold)
        )
        default_threshold = min(
            _THRESHOLD_MAX, max(_THRESHOLD_MIN, default_threshold)
        )
        threshold = st.slider(
            "Cosine-distance threshold",
            min_value=_THRESHOLD_MIN,
            max_value=_THRESHOLD_MAX,
            value=default_threshold,
            step=_THRESHOLD_STEP,
            help="A turn is flagged when its distance from the goal exceeds this.",
        )
        st.session_state["threshold"] = threshold
        monitor.set_threshold(threshold)

        # -- Embedder caption ------------------------------------------------ #
        st.divider()
        name = getattr(embedder, "name", "unknown")
        st.caption(f"Embedder: `{name}`")
        if _is_hashing(embedder):
            st.caption(
                "Pure-Python hashing fallback (offline). Install "
                "`sentence-transformers` for sharper semantic drift."
            )


# --------------------------------------------------------------------------- #
# Live drift panel (auto-refreshes itself)
# --------------------------------------------------------------------------- #
@st.fragment(run_every=_REFRESH_SECONDS)
def _live_drift_panel(session_id: str) -> None:
    """Auto-refreshing chart + banner + live corrective prompt.

    Re-runs every few seconds so turns captured by the background watcher appear
    without any interaction. Only read-only widgets live here, so auto-refresh
    never clobbers user input.
    """
    monitor = _get_monitor()
    try:
        ts = monitor.timeseries(session_id)
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.info(f"No drift data yet ({exc}).")
        return

    turns = ts.get("turns") or []
    threshold = float(ts.get("threshold", config.DEFAULT_THRESHOLD))

    if not turns:
        st.info(
            "Waiting for the first turn. Copy a message from your LLM chat "
            "(or use the tabs below) and it will appear here."
        )
        return

    anchor = ts.get("drift_from_anchor") or []
    reference = ts.get("drift_from_reference") or []
    events = set(ts.get("alignment_events") or [])

    last_anchor = anchor[-1] if anchor else 0.0
    last_reference = reference[-1] if reference else 0.0
    last_turn = turns[-1]
    drift_high = (
        last_turn in events or last_anchor > threshold or last_reference > threshold
    )

    # Headline metrics.
    m1, m2, m3 = st.columns(3)
    m1.metric("Turns captured", len(turns))
    m2.metric("Drift vs anchor", f"{last_anchor:.3f}")
    m3.metric("Status", "DRIFTING" if drift_high else "on track")

    # Chart.
    try:
        import pandas as pd

        frame = pd.DataFrame(
            {
                "drift_from_anchor": anchor,
                "drift_from_reference": reference,
                "threshold": [threshold] * len(turns),
            },
            index=pd.Index(turns, name="turn"),
        )
        st.line_chart(frame)
    except Exception:  # pragma: no cover - pandas-free fallback
        st.line_chart(
            {
                "drift_from_anchor": anchor,
                "drift_from_reference": reference,
                "threshold": [threshold] * len(turns),
            }
        )

    if drift_high:
        st.error(
            f"High drift on turn {last_turn}: anchor={last_anchor:.3f}, "
            f"reference={last_reference:.3f} (threshold {threshold:.2f}). "
            "Paste the corrective prompt below into your chat to re-align."
        )
        st.caption("Corrective prompt (use the copy button):")
        try:
            st.code(monitor.current_corrective_prompt(session_id), language="markdown")
        except Exception:  # pragma: no cover - defensive UI guard
            pass
    else:
        st.success(
            f"On track — latest drift anchor={last_anchor:.3f}, "
            f"reference={last_reference:.3f} (threshold {threshold:.2f})."
        )

    if events:
        st.caption(f"Drift-high turns: {sorted(events)}")


# --------------------------------------------------------------------------- #
# Turn-input tabs (manual fallback — auto-capture handles the common case)
# --------------------------------------------------------------------------- #
def _render_chat_tab(monitor: DriftMonitor, session_id: str) -> None:
    """Render the single-turn 'Add turn manually' input tab."""
    st.caption("Auto-capture handles this for you — this is a manual fallback.")
    role = st.radio("Role", options=["user", "assistant"], horizontal=True, key="chat_role")
    text = st.text_area("Turn text", value="", height=120, key="chat_text")
    if st.button("Add turn", key="add_turn_btn"):
        if not text.strip():
            st.warning("Enter some text before adding a turn.")
            return
        try:
            monitor.add_turn(session_id, role, text.strip())
        except Exception as exc:
            st.error(f"Could not add turn: {exc}")
            return
        st.rerun()


def _render_import_tab(monitor: DriftMonitor, session_id: str) -> None:
    """Render the 'Import transcript' tab: file upload + paste with format select."""
    fmt = st.selectbox(
        "Format", options=["auto", "json", "markdown", "text"], index=0, key="import_fmt"
    )
    uploaded = st.file_uploader(
        "Upload a transcript file", type=["json", "md", "txt"], key="import_file"
    )
    pasted = st.text_area(
        "…or paste transcript text",
        value="",
        height=160,
        key="import_paste",
        help="JSON list of messages, or lines prefixed 'User:' / 'Assistant:'.",
    )
    if st.button("Import & analyse", key="import_btn"):
        source = ""
        if uploaded is not None:
            try:
                source = uploaded.getvalue().decode("utf-8", errors="replace")
            except Exception as exc:
                st.error(f"Could not read uploaded file: {exc}")
                return
        elif pasted.strip():
            source = pasted
        if not source.strip():
            st.warning("Upload a file or paste transcript text first.")
            return
        turns = parse_transcript(source, fmt=fmt)
        if not turns:
            st.warning("No turns could be parsed from that input.")
            return
        try:
            result = monitor.ingest_transcript(session_id, turns)
        except Exception as exc:
            st.error(f"Could not import transcript: {exc}")
            return
        st.success(f"Imported {result['added']} turns ({result['alerts']} drift alerts).")
        st.rerun()


# --------------------------------------------------------------------------- #
# Corrective editor + goal state (normal flow — not auto-refreshed)
# --------------------------------------------------------------------------- #
def _render_corrective_editor(monitor: DriftMonitor, session_id: str) -> None:
    """Editable corrective prompt + checkpoint button (won't be clobbered live)."""
    with st.expander("Edit corrective prompt before pasting", expanded=False):
        try:
            prompt = monitor.current_corrective_prompt(session_id)
        except Exception as exc:  # pragma: no cover - defensive
            prompt = ""
            st.info(f"No corrective prompt yet ({exc}).")
        if prompt:
            st.text_area("Editable prompt", value=prompt, height=240, key="corrective_editor")
        if st.button("Mark checkpoint (re-anchor)", key="mark_checkpoint_btn"):
            try:
                monitor.mark_checkpoint(session_id)
                st.success("Checkpoint recorded — reference re-aligned to current context.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not mark checkpoint: {exc}")


def _render_goal_state(monitor: DriftMonitor, session_id: str) -> None:
    """Render the latest goal-state raw dict inside an expander as JSON."""
    with st.expander("Latest goal state", expanded=False):
        try:
            gs = monitor.latest_goal_state(session_id)
        except Exception as exc:  # pragma: no cover - defensive UI guard
            st.info(f"No goal state yet ({exc}).")
            return
        if gs is None:
            st.info("No goal state recorded yet.")
            return
        st.caption(f"Snapshot at turn: {gs.turn_snapshot}")
        st.json(gs.raw or {})


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    """Compose and render the full Streamlit app."""
    st.set_page_config(page_title="Context Drift Monitor", layout="wide")

    embedder = _build_embedder()
    monitor = _get_monitor()

    _resume_active_session(monitor)
    _render_sidebar(monitor, embedder)

    session_id = _active_session_id()
    if not session_id:
        st.title("Context Drift Monitor")
        st.info(
            "Create a session in the sidebar (project name + initial goal), then "
            "just chat with your LLM — every message you copy is captured "
            "automatically and the drift graph updates live."
        )
        return

    try:
        session = monitor.store.get_session(session_id)
    except Exception:  # pragma: no cover - defensive UI guard
        session = None
    if session is None:
        st.title("Context Drift Monitor")
        st.warning("The selected session no longer exists. Pick or create one.")
        _set_active_session(monitor, None)
        return

    st.title(session.project_name or "Context Drift Monitor")
    st.caption(f"Anchor goal: {session.anchor_goal}")
    if session.constraints:
        st.caption("Constraints: " + " · ".join(session.constraints))

    # The live, self-refreshing drift panel.
    _live_drift_panel(session_id)

    st.divider()
    tab_chat, tab_import = st.tabs(["Add turn manually", "Import transcript"])
    with tab_chat:
        _render_chat_tab(monitor, session_id)
    with tab_import:
        _render_import_tab(monitor, session_id)

    left, right = st.columns(2)
    with left:
        _render_corrective_editor(monitor, session_id)
    with right:
        _render_goal_state(monitor, session_id)


if __name__ == "__main__":
    main()
