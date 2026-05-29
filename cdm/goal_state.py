"""Heuristic goal-state extraction for the Context Drift Monitor.

This module distils a running conversation into a small, structured goal-state
dict with four keys (``core_goal``, ``constraints``, ``decisions``,
``current_focus``). It is intentionally **LLM-free**: everything is derived from
regular-expression mining and keyword frequency over the message history so it
runs fully offline and deterministically.

The ``core_goal`` is always pinned to the supplied anchor goal — it is the
"north star" and never changes. Constraints are the anchor constraints unioned
with limits mined from the user's own messages. Decisions are clauses that
signal a choice was made. ``current_focus`` summarises what the most recent
turns are about.

Only :mod:`re` and :mod:`collections` from the standard library are used.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List

from cdm.config import ROLLING_WINDOW
from cdm.models import Message

__all__ = ["extract_goal_state", "goal_state_text"]

# Cap on the number of decisions retained in a goal state.
_MAX_DECISIONS = 12
# Number of keywords used to summarise the current focus.
_FOCUS_KEYWORDS = 6
# Maximum length of the fallback current-focus phrase.
_FOCUS_FALLBACK_CHARS = 140

# --- English stopwords -------------------------------------------------------
# A pragmatic, inline stopword set (no external corpus / download required).
_STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are arent as at be
    because been before being below between both but by cant cannot could
    couldnt did didnt do does doesnt doing dont down during each few for from
    further had hadnt has hasnt have havent having he hed hell hes her here
    heres hers herself him himself his how hows i id ill im ive if in into is
    isnt it its itself lets me more most mustnt my myself no nor not of off on
    once only or other ought our ours ourselves out over own same shant she
    shed shell shes should shouldnt so some such than that thats the their
    theirs them themselves then there theres these they theyd theyll theyre
    theyve this those through to too under until up very was wasnt we wed well
    were weve werent what whats when whens where wheres which while who whos
    whom why whys with wont would wouldnt you youd youll youre youve your yours
    yourself yourselves
    just like get got make made want need also really thing things lot maybe
    okay ok yes yeah sure thanks thank please well going gonna let us use using
    used
    per good bad nice great fine kind sort way stuff pretty much many every even
    back around actually basically honestly probably anyway etc lots still able
    """.split()
)

# --- regex patterns ----------------------------------------------------------
# Sentence/clause splitter: break on sentence punctuation and common clause
# separators so we can scan one assertion at a time. The lookbehind/lookahead on
# `.!?` keep decimal points inside numbers intact (so "0.3 Nm" is not split into
# "0" and "3 Nm").
_CLAUSE_SPLIT = re.compile(
    r"(?<!\d)[.!?]+(?!\d)|[\n;]+|\bbut\b|\band\b|\bbecause\b", re.IGNORECASE
)

# Strong constraint markers: modal/obligation words that, on their own, make a
# clause a genuine constraint. Soft intentions ("should", "need to", "want to")
# are deliberately excluded — alone they fire on ordinary chatter ("we need to
# fix the snacks", "we should grab lunch") and pollute the mined constraints. A
# soft clause still qualifies if it carries a numeric limit (handled separately).
_STRONG_CONSTRAINT = re.compile(
    r"\b(?:must(?:\s+not)?|mustn'?t|shall(?:\s+not)?|"
    r"require[sd]?|requirement|"
    r"non[- ]?negotiable|mandatory|prohibited|forbidden|"
    r"cannot|can'?t)\b",
    re.IGNORECASE,
)

# Numeric limits with units, e.g. "5 kg", "10 L", "$200", "200 dollars",
# "under 5kg", "less than 3 meters". The presence of a number + (unit or a
# comparator nearby) is treated as a constraint.
_NUMERIC_UNIT = re.compile(
    r"\b\d+(?:\.\d+)?\s*"
    r"(?:kg|kilograms?|g|grams?|lb|lbs|pounds?|t|tonnes?|tons?|"
    r"mm|cm|m|km|meters?|metres?|inch(?:es)?|in|ft|feet|"
    r"ml|l|liters?|litres?|gal|gallons?|"
    r"hz|khz|mhz|ghz|"
    r"w|kw|v|a|mah|wh|kwh|"
    r"s|sec|seconds?|min|minutes?|hr|hrs|hours?|days?|weeks?|months?|"
    r"%|percent|fps|dpi|px|pixels?|"
    r"usd|eur|gbp|dollars?|euros?|pounds?)\b",
    re.IGNORECASE,
)
# Currency written with a leading symbol, e.g. "$200", "€1,500".
_CURRENCY = re.compile(r"[$€£]\s?\d[\d,]*(?:\.\d+)?")
# Bare comparator + number, e.g. "< 5", "under 10", "no more than 3".
_COMPARATOR_NUM = re.compile(
    r"\b(?:<|<=|>|>=|under|over|below|above|less\s+than|more\s+than|"
    r"no\s+more\s+than|no\s+less\s+than|at\s+most|at\s+least|up\s+to)\s*\d",
    re.IGNORECASE,
)

# Words that mark a decision-bearing clause.
_DECISION_KEYWORDS = re.compile(
    r"\b(?:chose|choose|chosen|decided|decide|deciding|"
    r"we\s+will\s+use|we'?ll\s+use|i\s+will\s+use|i'?ll\s+use|"
    r"selected|select|going\s+with|go\s+with|went\s+with|"
    r"added|adding|picked|pick|switch(?:ing|ed)?\s+to|"
    r"settled\s+on|opted\s+for|agreed\s+(?:on|to)|let'?s\s+(?:use|go\s+with)|"
    r"lock(?:ed|ing|s)?\s+in|decision\s+locked|finali[sz]ed?)\b",
    re.IGNORECASE,
)

# Tokeniser for keyword frequency.
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def _clauses(text: str) -> List[str]:
    """Split text into trimmed, non-empty clauses for assertion scanning."""
    parts = _CLAUSE_SPLIT.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def _is_constraint_clause(clause: str) -> bool:
    """Return True if a clause expresses a hard constraint or a numeric limit.

    A clause qualifies on a strong modal (must / shall / required /
    non-negotiable / cannot ...) OR on a numeric limit: a number with a unit, a
    currency amount, or a comparator + number. Soft intentions without a number
    are intentionally ignored so ordinary conversation is not mined as a
    constraint.
    """
    if _STRONG_CONSTRAINT.search(clause):
        return True
    if _NUMERIC_UNIT.search(clause):
        return True
    if _CURRENCY.search(clause):
        return True
    if _COMPARATOR_NUM.search(clause):
        return True
    return False


def _is_decision_clause(clause: str) -> bool:
    """Return True if a clause signals that a decision was made."""
    return bool(_DECISION_KEYWORDS.search(clause))


def _clean(text: str) -> str:
    """Collapse internal whitespace and trim a clause for storage."""
    return re.sub(r"\s+", " ", text).strip()


def _dedupe(items: List[str]) -> List[str]:
    """Order-stable, case-insensitive de-duplication of trimmed strings."""
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        cleaned = _clean(item)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _user_messages(messages: List[Message]) -> List[Message]:
    """Filter to user-authored turns (decisions/constraints come from the user)."""
    return [m for m in messages if (m.role or "").lower() == "user"]


def _mine_constraints(messages: List[Message]) -> List[str]:
    """Mine constraint clauses from user messages, in chronological order."""
    found: List[str] = []
    for msg in _user_messages(messages):
        for clause in _clauses(msg.text or ""):
            if _is_constraint_clause(clause):
                found.append(clause)
    return found


def _mine_decisions(messages: List[Message]) -> List[str]:
    """Mine decision clauses from user messages, most-recent-first, capped."""
    found: List[str] = []
    for msg in _user_messages(messages):
        for clause in _clauses(msg.text or ""):
            if _is_decision_clause(clause):
                found.append(clause)
    # Most-recent-first while preserving within-message order on reversal.
    found.reverse()
    deduped = _dedupe(found)
    return deduped[:_MAX_DECISIONS]


def _current_focus(messages: List[Message], window: int) -> str:
    """Summarise recent turns via keyword frequency, with a text fallback.

    Looks at the last ``window`` turns, counts meaningful (non-stopword) word
    tokens, and joins the most frequent. Falls back to the trimmed last user
    message (or last message) when no salient keywords are found.
    """
    if not messages:
        return ""

    recent = messages[-window:] if window > 0 else messages
    counts: Counter[str] = Counter()
    for msg in recent:
        for token in _WORD.findall((msg.text or "").lower()):
            # Drop apostrophes so contractions normalise onto the (apostrophe-free)
            # stopword set: "i'd" -> "id", "we'll" -> "well", "don't" -> "dont".
            word = token.replace("'", "").replace("’", "").strip("-")
            if len(word) < 3:
                continue
            if word in _STOPWORDS:
                continue
            counts[word] += 1

    if counts:
        keywords = [w for w, _ in counts.most_common(_FOCUS_KEYWORDS)]
        return ", ".join(keywords)

    # Fallback: last user message, else last message of any role.
    users = _user_messages(recent) or _user_messages(messages)
    source = users[-1] if users else messages[-1]
    text = _clean(source.text or "")
    if len(text) > _FOCUS_FALLBACK_CHARS:
        text = text[:_FOCUS_FALLBACK_CHARS].rstrip() + "…"
    return text


def extract_goal_state(
    messages: List[Message],
    anchor_goal: str,
    anchor_constraints: List[str],
    turn_snapshot: int,
    window: int = ROLLING_WINDOW,
) -> Dict[str, Any]:
    """Distil a conversation into a structured goal-state dict (heuristic, no LLM).

    Args:
        messages: Full ordered message history for the session.
        anchor_goal: The fixed original goal; becomes ``core_goal`` unchanged.
        anchor_constraints: Constraints declared up front for the session.
        turn_snapshot: The last turn index this snapshot summarises (carried by
            the caller for context; not embedded in the returned dict).
        window: Number of most-recent turns used to compute ``current_focus``.

    Returns:
        The canonical raw dict with keys ``core_goal`` (str), ``constraints``
        (list[str]), ``decisions`` (list[str], most-recent-first, capped), and
        ``current_focus`` (str, non-empty whenever any messages exist).
    """
    core_goal = (anchor_goal or "").strip()

    anchor = [_clean(c) for c in (anchor_constraints or []) if c and c.strip()]
    mined = _mine_constraints(messages or [])
    constraints = _dedupe(anchor + mined)

    decisions = _mine_decisions(messages or [])

    current_focus = _current_focus(messages or [], window)

    return {
        "core_goal": core_goal,
        "constraints": constraints,
        "decisions": decisions,
        "current_focus": current_focus,
    }


def goal_state_text(raw: Dict[str, Any]) -> str:
    """Render a goal-state dict to a stable string for reference embedding.

    Concatenates ``core_goal``, ``current_focus``, ``constraints`` and
    ``decisions`` in a fixed, readable order so the same goal state always
    yields the same text (and therefore the same embedding). Missing or empty
    keys are simply omitted.
    """
    raw = raw or {}
    parts: List[str] = []

    core_goal = _clean(str(raw.get("core_goal", "")))
    if core_goal:
        parts.append(f"Goal: {core_goal}")

    current_focus = _clean(str(raw.get("current_focus", "")))
    if current_focus:
        parts.append(f"Current focus: {current_focus}")

    constraints = raw.get("constraints") or []
    constraints = [_clean(str(c)) for c in constraints if str(c).strip()]
    if constraints:
        parts.append("Constraints: " + "; ".join(constraints))

    decisions = raw.get("decisions") or []
    decisions = [_clean(str(d)) for d in decisions if str(d).strip()]
    if decisions:
        parts.append("Decisions: " + "; ".join(decisions))

    return "\n".join(parts)
