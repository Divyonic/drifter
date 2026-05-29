"""SQLite persistence layer for the Context Drift Monitor.

Stores :class:`~cdm.models.Session`, :class:`~cdm.models.Message`,
:class:`~cdm.models.GoalState` and :class:`~cdm.models.DriftScore` rows.
Embeddings and the goal-state ``raw`` dict are serialised to JSON text.

The :class:`Store` opens its connection with ``check_same_thread=False`` and
enables WAL mode so it is safe to share across a Streamlit app's threads.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Union

from cdm import config
from cdm.models import DriftScore, GoalState, Message, Session

__all__ = ["Store"]


def _dumps(value: object) -> Optional[str]:
    """Serialise a JSON-able value to text, mapping ``None`` to SQL NULL."""
    if value is None:
        return None
    return json.dumps(value)


def _loads(value: Optional[str]) -> object:
    """Deserialise JSON text, mapping SQL NULL to ``None``."""
    if value is None:
        return None
    return json.loads(value)


class Store:
    """SQLite-backed persistence for sessions, messages, goal states and drift.

    Tables (``sessions``, ``messages``, ``goal_states``, ``drift_scores``) are
    created on construction. Embeddings are stored as JSON text. The connection
    is opened once with ``check_same_thread=False`` and WAL journalling.
    """

    def __init__(self, db_path: Union[str, Path] = config.DB_PATH) -> None:
        """Open (creating if needed) the SQLite database at ``db_path``.

        Ensures the parent directory exists, enables WAL mode and creates the
        schema. ``":memory:"`` is accepted for an ephemeral database.
        """
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    # -- schema ---------------------------------------------------------------

    def _create_tables(self) -> None:
        """Create all tables if they do not already exist."""
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id   TEXT PRIMARY KEY,
                    project_name TEXT NOT NULL,
                    anchor_goal  TEXT NOT NULL,
                    constraints  TEXT NOT NULL,
                    created_at   TEXT NOT NULL,
                    updated_at   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    session_id TEXT NOT NULL,
                    turn_id    INTEGER NOT NULL,
                    role       TEXT NOT NULL,
                    text       TEXT NOT NULL,
                    embedding  TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, turn_id)
                );

                CREATE TABLE IF NOT EXISTS goal_states (
                    session_id          TEXT NOT NULL,
                    turn_snapshot       INTEGER NOT NULL,
                    raw                 TEXT NOT NULL,
                    reference_embedding TEXT,
                    created_at          TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS drift_scores (
                    session_id           TEXT NOT NULL,
                    turn_id              INTEGER NOT NULL,
                    drift_from_reference REAL NOT NULL,
                    drift_from_anchor    REAL NOT NULL,
                    is_drift_high        INTEGER NOT NULL,
                    PRIMARY KEY (session_id, turn_id)
                );

                CREATE TABLE IF NOT EXISTS meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages (session_id, turn_id);
                CREATE INDEX IF NOT EXISTS idx_goal_states_session
                    ON goal_states (session_id, turn_snapshot);
                CREATE INDEX IF NOT EXISTS idx_drift_session
                    ON drift_scores (session_id, turn_id);
                """
            )

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    # -- cross-process meta (k/v) --------------------------------------------

    def set_meta(self, key: str, value: Optional[str]) -> None:
        """Upsert a small key/value pair shared across processes (app + watcher)."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Return the stored value for ``key`` (latest committed), or ``default``."""
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return default
        return row["value"]

    def set_active_session(self, session_id: Optional[str]) -> None:
        """Record which session the clipboard watcher should append turns to."""
        self.set_meta("active_session", session_id)

    def get_active_session(self) -> Optional[str]:
        """Return the session id the watcher is currently feeding, if any."""
        return self.get_meta("active_session")

    # -- sessions -------------------------------------------------------------

    def create_session(self, session: Session) -> None:
        """Insert a new session row."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO sessions "
                "(session_id, project_name, anchor_goal, constraints, "
                " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session.session_id,
                    session.project_name,
                    session.anchor_goal,
                    json.dumps(list(session.constraints)),
                    session.created_at,
                    session.updated_at,
                ),
            )

    def get_session(self, session_id: str) -> Optional[Session]:
        """Return the session with ``session_id`` or ``None`` if absent."""
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_session(row)

    def update_session(self, session: Session) -> None:
        """Overwrite an existing session's mutable fields (caller bumps timestamps)."""
        with self._conn:
            self._conn.execute(
                "UPDATE sessions SET project_name = ?, anchor_goal = ?, "
                "constraints = ?, created_at = ?, updated_at = ? "
                "WHERE session_id = ?",
                (
                    session.project_name,
                    session.anchor_goal,
                    json.dumps(list(session.constraints)),
                    session.created_at,
                    session.updated_at,
                    session.session_id,
                ),
            )

    def list_sessions(self) -> List[Session]:
        """Return all sessions, newest ``updated_at`` first."""
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC, session_id DESC"
        ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def delete_session(self, session_id: str) -> None:
        """Delete a session and cascade-delete its messages, goal states and drift."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM drift_scores WHERE session_id = ?", (session_id,)
            )
            self._conn.execute(
                "DELETE FROM goal_states WHERE session_id = ?", (session_id,)
            )
            self._conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            self._conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> Session:
        """Build a :class:`Session` from a sqlite row."""
        return Session(
            session_id=row["session_id"],
            project_name=row["project_name"],
            anchor_goal=row["anchor_goal"],
            constraints=list(json.loads(row["constraints"])),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # -- messages -------------------------------------------------------------

    def add_message(self, message: Message) -> None:
        """Insert (or replace on turn collision) a message row."""
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO messages "
                "(session_id, turn_id, role, text, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    message.session_id,
                    message.turn_id,
                    message.role,
                    message.text,
                    _dumps(message.embedding),
                    message.created_at,
                ),
            )

    def get_messages(self, session_id: str) -> List[Message]:
        """Return a session's messages ordered by ``turn_id``."""
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY turn_id ASC",
            (session_id,),
        ).fetchall()
        return [self._row_to_message(r) for r in rows]

    def next_turn_id(self, session_id: str) -> int:
        """Return ``max(turn_id) + 1`` for the session, or ``0`` if empty."""
        row = self._conn.execute(
            "SELECT MAX(turn_id) AS m FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None or row["m"] is None:
            return 0
        return int(row["m"]) + 1

    def delete_last_message(self, session_id: str) -> Optional[int]:
        """Delete the highest-``turn_id`` message (and its drift score).

        Returns the deleted ``turn_id``, or ``None`` if the session has no
        messages. Used to support 'regenerate'.
        """
        row = self._conn.execute(
            "SELECT MAX(turn_id) AS m FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None or row["m"] is None:
            return None
        turn_id = int(row["m"])
        with self._conn:
            self._conn.execute(
                "DELETE FROM messages WHERE session_id = ? AND turn_id = ?",
                (session_id, turn_id),
            )
            self._conn.execute(
                "DELETE FROM drift_scores WHERE session_id = ? AND turn_id = ?",
                (session_id, turn_id),
            )
        return turn_id

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> Message:
        """Build a :class:`Message` from a sqlite row."""
        return Message(
            turn_id=int(row["turn_id"]),
            session_id=row["session_id"],
            role=row["role"],
            text=row["text"],
            embedding=_loads(row["embedding"]),  # type: ignore[arg-type]
            created_at=row["created_at"],
        )

    # -- goal states ----------------------------------------------------------

    def add_goal_state(self, goal_state: GoalState) -> None:
        """Append a goal-state snapshot row."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO goal_states "
                "(session_id, turn_snapshot, raw, reference_embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    goal_state.session_id,
                    goal_state.turn_snapshot,
                    json.dumps(goal_state.raw),
                    _dumps(goal_state.reference_embedding),
                    goal_state.created_at,
                ),
            )

    def get_latest_goal_state(self, session_id: str) -> Optional[GoalState]:
        """Return the most recent goal state (highest ``turn_snapshot``)."""
        row = self._conn.execute(
            "SELECT * FROM goal_states WHERE session_id = ? "
            "ORDER BY turn_snapshot DESC, rowid DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_goal_state(row)

    def get_goal_states(self, session_id: str) -> List[GoalState]:
        """Return all goal states for a session ordered by ``turn_snapshot``."""
        rows = self._conn.execute(
            "SELECT * FROM goal_states WHERE session_id = ? "
            "ORDER BY turn_snapshot ASC, rowid ASC",
            (session_id,),
        ).fetchall()
        return [self._row_to_goal_state(r) for r in rows]

    @staticmethod
    def _row_to_goal_state(row: sqlite3.Row) -> GoalState:
        """Build a :class:`GoalState` from a sqlite row."""
        return GoalState(
            session_id=row["session_id"],
            turn_snapshot=int(row["turn_snapshot"]),
            raw=json.loads(row["raw"]),
            reference_embedding=_loads(row["reference_embedding"]),  # type: ignore[arg-type]
            created_at=row["created_at"],
        )

    # -- drift scores ---------------------------------------------------------

    def add_drift_score(self, score: DriftScore) -> None:
        """Upsert a drift score keyed on ``(session_id, turn_id)``."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO drift_scores "
                "(session_id, turn_id, drift_from_reference, drift_from_anchor, "
                " is_drift_high) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(session_id, turn_id) DO UPDATE SET "
                "  drift_from_reference = excluded.drift_from_reference, "
                "  drift_from_anchor = excluded.drift_from_anchor, "
                "  is_drift_high = excluded.is_drift_high",
                (
                    score.session_id,
                    score.turn_id,
                    float(score.drift_from_reference),
                    float(score.drift_from_anchor),
                    1 if score.is_drift_high else 0,
                ),
            )

    def get_drift_scores(self, session_id: str) -> List[DriftScore]:
        """Return all drift scores for a session ordered by ``turn_id``."""
        rows = self._conn.execute(
            "SELECT * FROM drift_scores WHERE session_id = ? ORDER BY turn_id ASC",
            (session_id,),
        ).fetchall()
        return [self._row_to_drift_score(r) for r in rows]

    @staticmethod
    def _row_to_drift_score(row: sqlite3.Row) -> DriftScore:
        """Build a :class:`DriftScore` from a sqlite row."""
        return DriftScore(
            turn_id=int(row["turn_id"]),
            session_id=row["session_id"],
            drift_from_reference=float(row["drift_from_reference"]),
            drift_from_anchor=float(row["drift_from_anchor"]),
            is_drift_high=bool(row["is_drift_high"]),
        )
