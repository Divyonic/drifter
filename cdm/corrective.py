"""Corrective-prompt rendering for the Context Drift Monitor.

When a conversation drifts off its anchor goal, the UI offers the user a
paste-ready "corrective prompt" that restates the inferred goal state and asks
the assistant to realign. :data:`CORRECTIVE_TEMPLATE` is the fixed template and
:func:`render_corrective_prompt` fills it from a goal-state raw dict.
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__ = ["CORRECTIVE_TEMPLATE", "render_corrective_prompt"]


CORRECTIVE_TEMPLATE: str = """You are acting as a strict alignment assistant. From now on, you must:
- Respect the following goal and constraints exactly:
  - Core goal: {core_goal}
  - Constraints:
{constraints}
  - Recent decisions:
{decisions}
  - Current focus: {current_focus}
- If any future message of mine contradicts these, prioritise them and say so.
- If my next message seems off-track, gently orient the conversation back to this goal.
First, summarise this corrected context in one short paragraph, then continue."""


def _bullet_list(items: Any) -> str:
    """Render ``items`` as a newline-joined ``- item`` list.

    Non-empty, trimmed entries become ``- item`` bullets. An empty or missing
    list (or one that contains only blanks) yields ``"(none specified)"``.
    """
    bullets: List[str] = []
    if isinstance(items, (list, tuple)):
        for item in items:
            text = str(item).strip()
            if text:
                bullets.append(f"- {text}")
    if not bullets:
        return "(none specified)"
    return "\n".join(bullets)


def render_corrective_prompt(goal_state_raw: Dict[str, Any]) -> str:
    """Fill :data:`CORRECTIVE_TEMPLATE` from a goal-state raw dict.

    Missing keys are tolerated. ``constraints`` and ``decisions`` are formatted
    as ``- item`` bullet lists (``"(none specified)"`` when empty). Returns
    paste-ready text.
    """
    raw: Dict[str, Any] = goal_state_raw or {}

    core_goal = str(raw.get("core_goal", "") or "").strip()
    current_focus = str(raw.get("current_focus", "") or "").strip()
    constraints = _bullet_list(raw.get("constraints"))
    decisions = _bullet_list(raw.get("decisions"))

    return CORRECTIVE_TEMPLATE.format(
        core_goal=core_goal,
        constraints=constraints,
        decisions=decisions,
        current_focus=current_focus,
    )
