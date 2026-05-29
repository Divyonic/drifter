# Context Drift Monitor — Module Interface Contract

This is the **binding contract** for every module. The shared dataclasses live in
`cdm/models.py` and config in `cdm/config.py` (already written — import, do not
redefine). Implement exactly the public signatures below. All vectors are
`list[float]`, unit-normalised by the embedder. Python 3.14, standard library +
`numpy` only for the core package (the Streamlit app may use `pandas`/`altair`,
which ship with `streamlit`). **`sentence-transformers` is OPTIONAL** and may be
absent — the package MUST import and run without it.

Code style: type hints, concise docstrings, no print() in library code (raise or
return). Every module gets a matching `tests/test_<module>.py` using `pytest`
(plain `assert`). Tests must pass offline with only the hashing embedder.

---

## cdm/embeddings.py  +  tests/test_embeddings.py

```python
class Embedder(Protocol):
    name: str          # e.g. "local:all-MiniLM-L6-v2" or "hashing:512"
    dim: int
    def encode(self, texts: list[str]) -> list[list[float]]: ...   # batch; returns unit-normalised vectors
    def encode_one(self, text: str) -> list[float]: ...

class HashingEmbedder:
    """Pure-Python + numpy, deterministic, zero-dependency fallback.
    Hash word unigrams AND char 3-5 grams into `dim` buckets with signed counts,
    apply sublinear tf (1+log) weighting, L2-normalise. Must give higher cosine
    similarity to paraphrases/topically-related text than to unrelated text."""
    def __init__(self, dim: int = config.HASHING_DIM): ...

class LocalEmbedder:
    """Wraps sentence-transformers (config.LOCAL_MODEL_NAME). Lazy-import inside
    __init__; raise ImportError/RuntimeError with a helpful message if the package
    or model is unavailable so the factory can fall back."""
    def __init__(self, model_name: str = config.LOCAL_MODEL_NAME): ...

def get_embedder(preference: str = config.EMBEDDER_PREFERENCE) -> Embedder:
    """auto -> try LocalEmbedder, fall back to HashingEmbedder.
    local -> LocalEmbedder (may raise). hashing -> HashingEmbedder."""

def cosine_similarity(a: list[float], b: list[float]) -> float   # in [-1, 1]; 0.0 if either is empty/zero
def cosine_distance(a: list[float], b: list[float]) -> float     # 1 - cosine_similarity, clamped to [0, 2]
```
Notes: cache the embedder build is NOT this module's job (monitor/app handle reuse).
Empty string -> zero vector -> distance handled gracefully (cosine_distance==1.0).

## cdm/drift.py  +  tests/test_drift.py

```python
class DriftEngine:
    def __init__(self, threshold: float = config.DEFAULT_THRESHOLD,
                 window: int = config.ROLLING_WINDOW): ...

    def score_turn(self, message_emb: list[float], anchor_emb: list[float],
                   reference_emb: list[float] | None,
                   turn_id: int, session_id: str) -> DriftScore:
        """drift_from_anchor = cosine_distance(message_emb, anchor_emb).
        drift_from_reference = cosine_distance(message_emb, reference_emb) if
        reference_emb else drift_from_anchor. is_drift_high = either distance >
        self.threshold."""

    @staticmethod
    def centroid(embeddings: list[list[float]]) -> list[float]:
        """Mean vector, L2-normalised. [] -> []."""

    def rolling_reference(self, embeddings: list[list[float]]) -> list[float]:
        """Centroid of the last `self.window` embeddings (unit-normalised)."""

    @staticmethod
    def smooth(values: list[float], window: int) -> list[float]:
        """Trailing moving average; window<=1 returns a copy unchanged; len preserved."""
```
Import cosine_distance from cdm.embeddings. Return real `DriftScore` instances.

## cdm/goal_state.py  +  tests/test_goal_state.py

```python
def extract_goal_state(messages: list[Message], anchor_goal: str,
                       anchor_constraints: list[str], turn_snapshot: int,
                       window: int = config.ROLLING_WINDOW) -> dict:
    """Heuristic, NO LLM/API. Returns the canonical raw dict with keys:
        core_goal: str          -> anchor_goal (kept stable; this is the north star)
        constraints: list[str]  -> anchor_constraints UNION constraints mined from
                                    user messages (regex: numeric limits like
                                    '< 5 kg', 'under 10 L', 'budget', 'must', 'should',
                                    'non-negotiable', 'require', 'no more than',
                                    'at most/least'). Deduped, order-stable, trimmed.
        decisions: list[str]    -> sentences/clauses signalling a decision
                                    ('chose', 'decided', 'we will use', 'selected',
                                    'going with', 'added', 'picked', 'switch to').
                                    Deduped, most-recent-first, cap ~12.
        current_focus: str      -> short phrase describing what the last `window`
                                    turns are about (e.g. salient noun phrase / most
                                    frequent meaningful keywords from recent turns,
                                    falling back to the last user message trimmed to
                                    ~140 chars). Never empty if any messages exist."""

def goal_state_text(raw: dict) -> str:
    """Canonical single string used to build the reference embedding. Concatenate
    core_goal, current_focus, constraints, decisions in a stable readable form."""
```
Use only `re`, `collections`. Reasonable English stopword list inline. Robust to
empty inputs (no messages -> core_goal=anchor_goal, others empty/best-effort).

## cdm/corrective.py  +  tests/test_corrective.py

```python
CORRECTIVE_TEMPLATE: str   # the exact template from the product spec, with
                           # {core_goal}, {constraints}, {decisions}, {current_focus} slots

def render_corrective_prompt(goal_state_raw: dict) -> str:
    """Fill the template from a goal-state raw dict. Format constraints/decisions
    as readable bullet lists ('- item' joined by newlines; '(none specified)' when
    empty). Missing keys tolerated. Returns paste-ready text."""
```
Template body (match the spec intent):
```
You are acting as a strict alignment assistant. From now on, you must:
- Respect the following goal and constraints exactly:
  - Core goal: {core_goal}
  - Constraints:
{constraints}
  - Recent decisions:
{decisions}
  - Current focus: {current_focus}
- If any future message of mine contradicts these, prioritise them and say so.
- If my next message seems off-track, gently orient the conversation back to this goal.
First, summarise this corrected context in one short paragraph, then continue.
```

## cdm/transcript.py  +  tests/test_transcript.py

```python
def parse_transcript(source: str, fmt: str = "auto") -> list[dict]:
    """Parse a conversation export into [{"role": "user"|"assistant", "text": str}, ...].
    `source` is raw text OR a file path (detect: if it exists as a path, read it).
    Supported fmt: "auto" | "json" | "markdown" | "text".
      - json: a list of {role/ speaker, content/ text/ message} objects, OR an
        object with a "messages"/"conversation" list. Map role synonyms
        (human/user -> user; ai/assistant/bot/model -> assistant). Unknown -> user.
      - markdown/text: lines/blocks prefixed 'User:'/'Assistant:' (case-insensitive,
        also '## User', 'Human:', 'AI:'). Accumulate multi-line turns until the next
        speaker marker. If no markers found, treat each non-empty line as alternating
        user/assistant starting with user.
    Trim whitespace; drop empty turns. Never raise on malformed input — return best
    effort (possibly [])."""
```

## cdm/storage.py  +  tests/test_storage.py

```python
class Store:
    """SQLite persistence. Embeddings stored as JSON text. Creates tables on init.
    Thread-safe enough for Streamlit: open connections with check_same_thread=False
    OR open a fresh connection per call. WAL mode."""
    def __init__(self, db_path: str | Path = config.DB_PATH): ...   # ensure parent dir exists

    # sessions
    def create_session(self, session: Session) -> None: ...
    def get_session(self, session_id: str) -> Session | None: ...
    def update_session(self, session: Session) -> None: ...         # bump updated_at handled by caller
    def list_sessions(self) -> list[Session]: ...                   # newest updated first
    def delete_session(self, session_id: str) -> None: ...          # cascade messages/goal_states/drift

    # messages
    def add_message(self, message: Message) -> None: ...
    def get_messages(self, session_id: str) -> list[Message]: ...   # ordered by turn_id
    def next_turn_id(self, session_id: str) -> int: ...             # max(turn_id)+1 or 0

    # goal states
    def add_goal_state(self, goal_state: GoalState) -> None: ...
    def get_latest_goal_state(self, session_id: str) -> GoalState | None: ...
    def get_goal_states(self, session_id: str) -> list[GoalState]: ...  # ordered by turn_snapshot

    # drift scores
    def add_drift_score(self, score: DriftScore) -> None: ...        # upsert on (session_id, turn_id)
    def get_drift_scores(self, session_id: str) -> list[DriftScore]: ...  # ordered by turn_id
```
Tables: sessions, messages, goal_states, drift_scores. Tests use a tmp_path db.

## cdm/monitor.py  +  tests/test_monitor.py  (the orchestrator — ties everything together)

```python
class DriftMonitor:
    def __init__(self, store: Store | None = None, embedder: Embedder | None = None,
                 threshold: float = config.DEFAULT_THRESHOLD,
                 update_every: int = config.UPDATE_EVERY,
                 window: int = config.ROLLING_WINDOW): ...
        # default store=Store(); embedder=get_embedder(); build a DriftEngine.

    def start_session(self, project_name: str, initial_goal: str,
                      constraints: list[str] | None = None) -> Session:
        """Create+persist a Session (uuid4 hex id, ISO timestamps from
        datetime.now().isoformat() — note: scripts elsewhere can't call now(), but
        this RUNTIME module may). Persist an initial GoalState (turn_snapshot=-1)
        whose reference_embedding = embed(anchor_goal). Return the Session."""

    def add_turn(self, session_id: str, role: str, text: str) -> dict:
        """Embed text; store Message with the next turn_id; compute DriftScore vs
        anchor_emb (embed of session.anchor_goal, cache ok) and the latest
        goal-state reference_embedding; persist the score. Every `update_every`
        user/assistant turns (or when none exists), regenerate the GoalState via
        extract_goal_state over recent messages and persist it (re-embedding its
        goal_state_text). Returns:
          {"message": Message, "drift": DriftScore, "goal_state": GoalState,
           "alert": bool, "corrective_prompt": str|None}  # prompt set when alert."""

    def ingest_transcript(self, session_id: str, turns: list[dict]) -> dict:
        """Bulk add_turn over [{role,text},...]. Returns
        {"added": int, "alerts": int, "final_goal_state": GoalState}."""

    def timeseries(self, session_id: str) -> dict:
        """{"turns": [...], "roles": [...], "texts": [...] (trimmed),
            "drift_from_anchor": [...], "drift_from_reference": [...],
            "threshold": float, "alignment_events": [turn_id,...]}.
        Apply config.SMOOTHING_WINDOW smoothing to both drift series."""

    def current_corrective_prompt(self, session_id: str) -> str:
        """render_corrective_prompt(latest goal_state.raw)."""

    def latest_goal_state(self, session_id: str) -> GoalState | None: ...
    def set_threshold(self, value: float) -> None: ...               # updates engine + self
    def mark_checkpoint(self, session_id: str) -> None:
        """Record an alignment event (e.g. add a GoalState revision now). Used by UI
        after the user pastes a corrective prompt."""
```
This module may use datetime.now()/uuid because it runs at app runtime (NOT inside
a Workflow script). Import from cdm.models, cdm.storage, cdm.embeddings,
cdm.drift, cdm.goal_state, cdm.corrective.

## app.py  (Streamlit UI, repo root)

A single-file Streamlit app built ONLY on `DriftMonitor`'s public API. Requirements:
- Sidebar: "New session" form (project name, initial goal textarea, constraints — one
  per line); a selector of existing sessions (st.session_state holds the active
  DriftMonitor + session_id); threshold slider (0.30–0.95, default from config) wired
  to set_threshold; a small caption showing the active embedder name + a note when
  running on the hashing fallback.
- Main area, two ways to feed turns (tabs):
    1. "Chat / paste turn": a role radio (user/assistant) + text area + "Add turn"
       button calling add_turn; show the returned drift values.
    2. "Import transcript": file_uploader (.json/.md/.txt) AND a paste textarea +
       format selector -> parse_transcript -> ingest_transcript.
- Live line chart (st.line_chart over a DataFrame) of drift_from_anchor &
  drift_from_reference vs turn, plus a horizontal threshold reference (add a
  constant 'threshold' column). Mark alignment_events.
- Warning banner (st.error/st.warning) when the latest turn is drift-high.
- A corrective-prompt panel: st.code(prompt) (gives a copy button) + an editable
  st.text_area + a "Mark checkpoint" button (mark_checkpoint).
- A goal-state expander showing the latest goal-state JSON (st.json).
- Cache the embedder with @st.cache_resource so it isn't rebuilt each rerun.
- Must run with: `streamlit run app.py`. Never crash on empty/new session.

## Demo assets (repo root)
- sample_transcript.json — a ~14-turn conversation that starts on a clear goal
  (e.g. "design a pan-tilt EO/IR mount under 5 kg") and visibly drifts (into
  tangents like office snacks / unrelated topics) so the drift graph clearly rises.
- run.sh — create venv, pip install -r requirements.txt, streamlit run app.py.
