"""Runtime orchestrator for the Context Drift Monitor.

:class:`DriftMonitor` ties together storage, embeddings, drift scoring, goal-state
extraction and corrective-prompt rendering into a single object the Streamlit app
(and tests) drive. It owns a :class:`~cdm.storage.Store`, an
:class:`~cdm.embeddings.Embedder` and a :class:`~cdm.drift.DriftEngine`.

Unlike the offline Workflow scripts, this module runs at *app runtime* so it may
mint identifiers with :func:`uuid.uuid4` and timestamps with
:func:`datetime.datetime.now`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from cdm import config
from cdm.corrective import render_corrective_prompt
from cdm.drift import (
    DriftEngine,
    baseline_stats,
    biggest_jump,
    cusum_changepoint,
    forecast_cross,
)
from cdm.embeddings import Embedder, get_embedder
from cdm.goal_state import extract_goal_state, goal_state_text
from cdm.models import DriftScore, GoalState, Message, Session
from cdm.storage import Store

__all__ = ["DriftMonitor"]

# Number of characters retained per turn in :meth:`DriftMonitor.timeseries`.
_TEXT_TRIM = 80
# turn_snapshot value reserved for the very first (anchor) goal state.
_INITIAL_SNAPSHOT = -1


def _now() -> str:
    """Return the current local time as an ISO-8601 string."""
    return datetime.now().isoformat()


class DriftMonitor:
    """Drive a monitored conversation end to end.

    The monitor persists every turn, scores its drift against both the fixed
    anchor goal and a rolling goal-state reference, periodically re-derives the
    structured goal state, and surfaces a paste-ready corrective prompt whenever a
    turn drifts high.
    """

    def __init__(
        self,
        store: Optional[Store] = None,
        embedder: Optional[Embedder] = None,
        threshold: Optional[float] = None,
        update_every: int = config.UPDATE_EVERY,
        window: int = config.ROLLING_WINDOW,
    ) -> None:
        """Build a monitor.

        Args:
            store: Persistence backend. Defaults to a :class:`Store` at the
                configured database path.
            embedder: Embedding backend. Defaults to :func:`get_embedder` (auto:
                local if available, hashing otherwise).
            threshold: Cosine-distance value above which a turn is flagged. When
                ``None`` (the default) the active embedder's ``suggested_threshold``
                is used, since the hashing fallback and neural embeddings live on
                different distance scales.
            update_every: Regenerate the goal state every this many turns.
            window: Rolling window used for goal extraction and reference
                centroids.
        """
        self.store: Store = store if store is not None else Store()
        self.embedder: Embedder = embedder if embedder is not None else get_embedder()
        if threshold is None:
            threshold = getattr(
                self.embedder, "suggested_threshold", config.DEFAULT_THRESHOLD
            )
        self.threshold: float = float(threshold)
        self.update_every: int = max(1, int(update_every))
        self.window: int = max(1, int(window))
        self.engine = DriftEngine(threshold=self.threshold, window=self.window)
        # Cache of anchor-goal embeddings keyed by session id (avoids re-embedding
        # the immutable anchor on every turn).
        self._anchor_cache: Dict[str, List[float]] = {}

    # -- session lifecycle ----------------------------------------------------

    def start_session(
        self,
        project_name: str,
        initial_goal: str,
        constraints: Optional[List[str]] = None,
    ) -> Session:
        """Create and persist a new monitored session.

        A fresh :class:`Session` is stored with a uuid4 hex id and ISO
        timestamps, followed by an initial :class:`GoalState`
        (``turn_snapshot=-1``) whose ``reference_embedding`` is the embedding of
        the anchor goal.

        Args:
            project_name: Human-readable label for the conversation.
            initial_goal: The anchor goal (the conversation's north star).
            constraints: Optional up-front constraints.

        Returns:
            The persisted :class:`Session`.
        """
        constraints = list(constraints or [])
        now = _now()
        session = Session(
            session_id=uuid4().hex,
            project_name=project_name,
            anchor_goal=initial_goal,
            constraints=constraints,
            created_at=now,
            updated_at=now,
        )
        self.store.create_session(session)

        anchor_emb = self._anchor_embedding(session)
        raw = extract_goal_state(
            messages=[],
            anchor_goal=initial_goal,
            anchor_constraints=constraints,
            turn_snapshot=_INITIAL_SNAPSHOT,
            window=self.window,
        )
        initial_state = GoalState(
            session_id=session.session_id,
            turn_snapshot=_INITIAL_SNAPSHOT,
            raw=raw,
            reference_embedding=anchor_emb,
            created_at=now,
        )
        self.store.add_goal_state(initial_state)
        return session

    # -- turns ----------------------------------------------------------------

    def add_turn(self, session_id: str, role: str, text: str) -> Dict[str, Any]:
        """Add, embed, store and score a single conversation turn.

        Args:
            session_id: Target session id.
            role: ``"user"`` or ``"assistant"`` (any other value is stored as-is).
            text: The turn's text.

        Returns:
            A dict with keys ``message`` (:class:`Message`), ``drift``
            (:class:`DriftScore`), ``goal_state`` (:class:`GoalState`), ``alert``
            (bool) and ``corrective_prompt`` (str when alerting, else ``None``).

        Raises:
            ValueError: If ``session_id`` does not name an existing session.
        """
        session = self.store.get_session(session_id)
        if session is None:
            raise ValueError(f"Unknown session_id: {session_id!r}")

        now = _now()
        turn_id = self.store.next_turn_id(session_id)
        embedding = self.embedder.encode_one(text)
        message = Message(
            turn_id=turn_id,
            session_id=session_id,
            role=role,
            text=text,
            embedding=embedding,
            created_at=now,
        )
        self.store.add_message(message)

        anchor_emb = self._anchor_embedding(session)
        latest_state = self.store.get_latest_goal_state(session_id)
        reference_emb = latest_state.reference_embedding if latest_state else None

        drift = self.engine.score_turn(
            message_emb=embedding,
            anchor_emb=anchor_emb,
            reference_emb=reference_emb,
            turn_id=turn_id,
            session_id=session_id,
        )
        self.store.add_drift_score(drift)

        # Periodically regenerate the structured goal state from recent turns.
        goal_state = self._maybe_update_goal_state(session, turn_id, now, latest_state)

        # Keep the session's updated_at fresh.
        session.updated_at = now
        self.store.update_session(session)

        alert = bool(drift.is_drift_high)
        corrective_prompt = (
            render_corrective_prompt(goal_state.raw) if alert and goal_state else None
        )
        return {
            "message": message,
            "drift": drift,
            "goal_state": goal_state,
            "alert": alert,
            "corrective_prompt": corrective_prompt,
        }

    def ingest_transcript(
        self, session_id: str, turns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Bulk-add a list of ``{"role", "text"}`` turns to a session.

        Args:
            session_id: Target session id.
            turns: Ordered list of dicts with ``role`` and ``text`` keys.

        Returns:
            A dict with ``added`` (int), ``alerts`` (int) and ``final_goal_state``
            (:class:`GoalState` or ``None`` if the session has no goal state).
        """
        added = 0
        alerts = 0
        for turn in turns or []:
            text = str(turn.get("text", "") if isinstance(turn, dict) else "")
            if not text.strip():
                continue
            role = str(turn.get("role", "user")) if isinstance(turn, dict) else "user"
            result = self.add_turn(session_id, role, text)
            added += 1
            if result["alert"]:
                alerts += 1
        return {
            "added": added,
            "alerts": alerts,
            "final_goal_state": self.store.get_latest_goal_state(session_id),
        }

    # -- read-side ------------------------------------------------------------

    def timeseries(self, session_id: str) -> Dict[str, Any]:
        """Return the per-turn drift series for plotting.

        Both drift series are smoothed with a trailing moving average of size
        :data:`cdm.config.SMOOTHING_WINDOW`.

        Args:
            session_id: Target session id.

        Returns:
            A dict with parallel lists ``turns``, ``roles``, ``texts`` (trimmed),
            ``drift_from_anchor`` and ``drift_from_reference`` (both smoothed),
            plus ``threshold`` (float) and ``alignment_events`` (list of
            drift-high turn ids).
        """
        messages = self.store.get_messages(session_id)
        scores = self.store.get_drift_scores(session_id)
        by_turn = {s.turn_id: s for s in scores}

        turns: List[int] = []
        roles: List[str] = []
        texts: List[str] = []
        anchor_series: List[float] = []
        reference_series: List[float] = []
        alignment_events: List[int] = []

        for msg in messages:
            score = by_turn.get(msg.turn_id)
            turns.append(msg.turn_id)
            roles.append(msg.role)
            text = (msg.text or "").strip()
            if len(text) > _TEXT_TRIM:
                text = text[:_TEXT_TRIM].rstrip() + "…"
            texts.append(text)
            if score is not None:
                anchor_series.append(score.drift_from_anchor)
                reference_series.append(score.drift_from_reference)
                if score.is_drift_high:
                    alignment_events.append(msg.turn_id)
            else:
                anchor_series.append(0.0)
                reference_series.append(0.0)

        smoothing = config.SMOOTHING_WINDOW
        anchor_s = self.engine.smooth(anchor_series, smoothing)
        reference_s = self.engine.smooth(reference_series, smoothing)

        # Self-calibrating analytics. Changepoint/attribution run on the RAW series
        # (smoothing flattens the baseline and the step, distorting detection);
        # the forecast uses the smoothed series for a stable trend.
        mean, std = baseline_stats(anchor_series, k=4)
        cp_idx = cusum_changepoint(anchor_series, mean, std)
        jump_idx = biggest_jump(anchor_series)
        forecast = forecast_cross(anchor_s, self.threshold)

        def _turn_at(idx):
            return turns[idx] if (idx is not None and 0 <= idx < len(turns)) else None

        return {
            "turns": turns,
            "roles": roles,
            "texts": texts,
            "drift_from_anchor": anchor_s,
            "drift_from_reference": reference_s,
            "threshold": self.threshold,
            "alignment_events": alignment_events,
            # analytics
            "baseline_mean": mean,
            "baseline_std": std,
            "changepoint_turn": _turn_at(cp_idx),
            "attribution_turn": _turn_at(jump_idx),
            "forecast_turns": forecast,
            "forecast_will_cross": bool(forecast is not None and forecast > 0),
        }

    def is_drifting_calibrated(
        self, session_id: str, k_turn: float = 1.0, min_history: int = 4
    ) -> Dict[str, Any]:
        """Self-calibrating drift verdict for the *latest* turn of a session.

        The raw per-turn :attr:`DriftScore.is_drift_high` flag compares a single
        turn's cosine distance against a fixed absolute threshold. That fires on
        any lexically-divergent-but-on-goal prompt (a short "add a --header flag"
        scores far from a goal sentence), which makes it unusable as a per-prompt
        alarm — especially with the hashing embedder, where on-goal and off-goal
        turns occupy the same distance band.

        This instead learns what "normal" drift looks like for *this* conversation
        (:func:`baseline_stats`) and reports drift only when the series has
        *sustainedly* departed that baseline (:func:`cusum_changepoint`) **and**
        the most recent turn is itself still elevated above it — so returning to
        the goal clears the alert instead of nagging on every later turn. It is
        the same detector :meth:`timeseries` and the eval harness use, so the
        per-prompt verdict is consistent with the chart the user sees.

        Args:
            session_id: Target session id.
            k_turn: How many baseline std-devs above the baseline mean the latest
                turn must sit to count as "still drifting".
            min_history: Minimum number of scored turns required before any
                verdict is given; below this the conversation is too young to
                have a meaningful baseline, so it is reported as on-track.

        Returns:
            ``{"high": bool, "drift": float, "changepoint_turn": int | None,
            "baseline_mean": float, "baseline_std": float}``.
        """
        scores = sorted(
            self.store.get_drift_scores(session_id), key=lambda s: s.turn_id
        )
        series = [float(s.drift_from_anchor) for s in scores]
        out: Dict[str, Any] = {
            "high": False,
            "drift": series[-1] if series else 0.0,
            "changepoint_turn": None,
            "baseline_mean": 0.0,
            "baseline_std": 0.0,
        }
        if len(series) < min_history:
            return out  # too early to judge drift against a baseline
        mean, std = baseline_stats(series, k=4)
        cp = cusum_changepoint(series, mean, std)
        last = series[-1]
        out.update(baseline_mean=mean, baseline_std=std, changepoint_turn=cp)
        out["high"] = cp is not None and last > mean + k_turn * std
        return out

    def current_corrective_prompt(self, session_id: str) -> str:
        """Render the corrective prompt from the latest goal state.

        Args:
            session_id: Target session id.

        Returns:
            Paste-ready corrective-prompt text. If no goal state exists yet, an
            empty goal-state dict is rendered.
        """
        latest = self.store.get_latest_goal_state(session_id)
        raw = latest.raw if latest is not None else {}
        # Threshold-aware: stricter threshold -> tighter re-anchor wording.
        return render_corrective_prompt(raw, threshold=self.threshold)

    def latest_goal_state(self, session_id: str) -> Optional[GoalState]:
        """Return the most recent goal state for a session, or ``None``."""
        return self.store.get_latest_goal_state(session_id)

    def remove_last_turn(self, session_id: str) -> Optional[int]:
        """Drop the most recent turn + its drift score (used for 'regenerate')."""
        return self.store.delete_last_message(session_id)

    # -- controls -------------------------------------------------------------

    def set_threshold(self, value: float) -> None:
        """Update the drift threshold on both the monitor and its engine.

        Existing stored scores are not re-evaluated; the new threshold applies to
        subsequently added turns and is reflected in :meth:`timeseries`.

        Args:
            value: The new cosine-distance threshold.
        """
        self.threshold = float(value)
        self.engine.threshold = float(value)

    def mark_checkpoint(self, session_id: str) -> None:
        """Record an alignment checkpoint (a fresh goal-state revision).

        Called by the UI after the user pastes a corrective prompt: it re-derives
        the goal state from the full history and persists it pinned to the current
        last turn, so the rolling reference re-aligns to the corrected context.

        Args:
            session_id: Target session id.
        """
        session = self.store.get_session(session_id)
        if session is None:
            raise ValueError(f"Unknown session_id: {session_id!r}")
        messages = self.store.get_messages(session_id)
        last_turn = messages[-1].turn_id if messages else _INITIAL_SNAPSHOT
        self._regenerate_goal_state(session, messages, last_turn, _now())

    # -- internals ------------------------------------------------------------

    def _anchor_embedding(self, session: Session) -> List[float]:
        """Return (and cache) the embedding of a session's immutable anchor goal."""
        cached = self._anchor_cache.get(session.session_id)
        if cached is not None:
            return cached
        emb = self.embedder.encode_one(session.anchor_goal)
        self._anchor_cache[session.session_id] = emb
        return emb

    def _maybe_update_goal_state(
        self,
        session: Session,
        turn_id: int,
        now: str,
        latest_state: Optional[GoalState],
    ) -> GoalState:
        """Regenerate the goal state on the update cadence; return the current one.

        The goal state is re-derived when there is no non-initial snapshot yet, or
        every ``update_every`` turns. Otherwise the existing latest goal state is
        returned unchanged.
        """
        message_count = turn_id + 1
        only_initial = latest_state is None or latest_state.turn_snapshot < 0
        due = message_count % self.update_every == 0
        if only_initial or due:
            messages = self.store.get_messages(session.session_id)
            return self._regenerate_goal_state(session, messages, turn_id, now)
        # latest_state is guaranteed non-None here (only_initial would be True).
        return latest_state  # type: ignore[return-value]

    def _regenerate_goal_state(
        self,
        session: Session,
        messages: List[Message],
        turn_snapshot: int,
        now: str,
    ) -> GoalState:
        """Extract, embed and persist a fresh goal state; return it."""
        raw = extract_goal_state(
            messages=messages,
            anchor_goal=session.anchor_goal,
            anchor_constraints=list(session.constraints),
            turn_snapshot=turn_snapshot,
            window=self.window,
        )
        text = goal_state_text(raw)
        reference_emb = self.embedder.encode_one(text) if text else None
        goal_state = GoalState(
            session_id=session.session_id,
            turn_snapshot=turn_snapshot,
            raw=raw,
            reference_embedding=reference_emb,
            created_at=now,
        )
        self.store.add_goal_state(goal_state)
        return goal_state
