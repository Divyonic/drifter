"""Core data model for the Context Drift Monitor.

These dataclasses are the single source of truth shared by every module
(storage, embeddings, drift, goal_state, monitor, app). Do not change field
names or types without updating CONTRACT.md and every consumer.

Embeddings are carried in memory as ``list[float]`` (unit-normalised by the
embedder) and serialised to JSON text when persisted.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

__all__ = ["Session", "Message", "GoalState", "DriftScore", "to_dict"]


@dataclass
class Session:
    """A monitored conversation with a fixed anchor goal."""

    session_id: str
    project_name: str
    anchor_goal: str
    constraints: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Message:
    """A single turn (user or assistant)."""

    turn_id: int                      # 0-based, monotonically increasing within a session
    session_id: str
    role: str                         # "user" | "assistant"
    text: str
    embedding: Optional[List[float]] = None
    created_at: str = ""


@dataclass
class GoalState:
    """A snapshot of the inferred user goal at a point in the conversation.

    ``raw`` always carries the four keys produced by goal extraction:
        core_goal: str
        constraints: list[str]
        decisions: list[str]
        current_focus: str
    """

    session_id: str
    turn_snapshot: int                # last turn index this snapshot summarises
    raw: Dict[str, Any] = field(default_factory=dict)
    reference_embedding: Optional[List[float]] = None
    created_at: str = ""


@dataclass
class DriftScore:
    """Per-turn drift measurement."""

    turn_id: int
    session_id: str
    drift_from_reference: float       # cosine distance: message vs current goal-state reference
    drift_from_anchor: float          # cosine distance: message vs original anchor goal
    is_drift_high: bool


def to_dict(obj: Any) -> Dict[str, Any]:
    """Convenience wrapper around :func:`dataclasses.asdict`."""
    return asdict(obj)
