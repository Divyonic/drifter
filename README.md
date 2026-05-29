# Drifter — Context Drift Monitor

A **native desktop app** that watches your LLM conversation drift away from its
original goal **in real time**, and re-aligns it with one click when it does.

Over a long session an assistant slowly *forgets the goal* — it loosens early
constraints, accumulates small deviations, and ends up answering something adjacent
to, but no longer aligned with, what you actually wanted. Drifter chats with your LLM
in-app, scores every turn's drift, keeps a running estimate of your
goal/constraints/decisions, and surfaces a paste-ready corrective prompt the moment
drift crosses the line.

**Local by design.** Sessions live in SQLite on your machine, API keys are stored
locally, and the drift engine runs fully offline. API calls go **straight from your PC
to the provider** — there is no server in between.

---

## Install

```bash
pip install "git+https://github.com/Divyonic/drifter.git#egg=drifter[llm]"
drifter
```

`drifter` opens the desktop app. The `[llm]` extra pulls the chat SDKs (Claude, Gemini,
OpenAI); drop it if you only want offline monitoring.

### From a checkout

```bash
git clone https://github.com/Divyonic/drifter.git && cd drifter
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[llm]"
drifter
```

---

## How to use it

1. **Launch** → Drifter asks *which session you want to continue today*. Pick a past
   session, start a **New session** (project + goal + constraints), or **Import** a
   past chat transcript (JSON / `User:`-`Assistant:` markdown) to resume monitoring it.
2. **Settings** → choose your provider (Claude / Gemini / OpenAI), model, and paste an
   API key (stored locally only).
3. **Chat** in the app. Every message and reply is scored; the **drift chart updates
   live** (orange = drift vs your original goal, grey dashed = vs the rolling
   reference, red dotted = threshold).
4. When a turn crosses the threshold the status flips to **DRIFTING** and a
   **corrective prompt** appears. Hit **Send to re-align** (or **Copy**). With
   *Auto re-align* on, Drifter folds the corrective prompt into the next request for you.

Want to monitor a chat you're having **in a browser** (e.g. Gemini on the web) instead
of in-app? Tick **"Also capture clipboard"** — anything you copy (Cmd/Ctrl+C) is added
as a turn.

### Commands

```bash
drifter          # native desktop app (default)
drifter web      # optional Streamlit browser app (localhost-only)
drifter watch    # clipboard capture watcher on its own
drifter version
```

---

## How it works

```
your message ─▶ embed ─▶ drift score ─┬─▶ live chart (vs anchor & rolling reference)
       │                               └─▶ threshold crossed? ─▶ corrective prompt ─▶ (auto) re-align next call
       └─▶ send to provider API ─▶ reply ─▶ embed ─▶ drift score ─▶ chart …
```

- **Anchor vs reference.** `drift_from_anchor` is the cosine distance from a turn to
  your *original* goal (never changes); `drift_from_reference` is the distance to the
  current goal-state snapshot, re-derived every few turns. A turn is flagged when
  either exceeds the threshold.
- **Goal-state extraction is heuristic and offline.** Constraints are mined from your
  messages (strong modals like *must / required / non-negotiable*, plus numeric limits
  like `< 5 kg`, `10 L`, `$200`); decisions from choice phrases (*chose, decided, go
  with, lock in…*); `current_focus` from recent keywords. Your goal is kept verbatim.
- **Embeddings.** Default is a pure-Python, offline **hashing** embedder (lexical
  drift). For semantic drift install the optional neural backend:
  `pip install sentence-transformers` (auto-detected; needs wheels for your Python).
- **Smoothing.** The chart shows a short trailing moving average — the *trend* is the
  signal; raw per-turn scores still drive the immediate flag.

---

## Configuration

Environment variables (defaults shown):

| Var | Default | Meaning |
|---|---|---|
| `CDM_DB_PATH` | `~/.context-drift-monitor/cdm.db` | SQLite database |
| `CDM_THRESHOLD` | `0.65` | Threshold for neural embeddings |
| `CDM_HASHING_THRESHOLD` | `0.80` | Threshold for the hashing fallback |
| `CDM_UPDATE_EVERY` | `5` | Re-derive the goal state every N turns |
| `CDM_WINDOW` | `10` | Recent-turn window for goal extraction |
| `CDM_SMOOTHING` | `3` | Trailing moving-average window for the chart |
| `CDM_EMBEDDER` | `auto` | `auto` / `local` / `hashing` |

API keys live in `~/.context-drift-monitor/credentials.json` (chmod 600), or set
`ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `OPENAI_API_KEY`.

---

## Use the engine as a library

The app is a thin shell over `cdm.monitor.DriftMonitor`:

```python
from cdm.monitor import DriftMonitor

mon = DriftMonitor()                       # offline by default
s = mon.start_session("My project", "design a 5 kg gimbal mount", ["< 5 kg"])
res = mon.add_turn(s.session_id, "user", "actually, what's for lunch?")
if res["alert"]:
    print(res["corrective_prompt"])        # paste-ready re-alignment prompt
ts = mon.timeseries(s.session_id)          # series for plotting
```

---

## Project layout

```
cdm/
  models.py       dataclasses: Session, Message, GoalState, DriftScore
  config.py       env-overridable settings
  embeddings.py   Embedder protocol, HashingEmbedder, LocalEmbedder, cosine math
  drift.py        DriftEngine: cosine distance, rolling reference, smoothing
  goal_state.py   heuristic goal/constraint/decision/focus extraction
  corrective.py   corrective-prompt template + renderer
  transcript.py   transcript parsing (JSON / markdown / text)
  storage.py      SQLite persistence + cross-process meta
  monitor.py      DriftMonitor — the orchestrator
  watcher.py      clipboard auto-capture (background process)
  llm.py          Claude / Gemini / OpenAI adapters + local key storage
  desktop.py      native PySide6 + pyqtgraph desktop app
  cli.py          the `drifter` command
  app.py          optional Streamlit browser app (`drifter web`)
tests/            pytest suite (offline, deterministic)
sample_transcript.json   14-turn demo that starts on-goal and drifts
pyproject.toml    packaging + `drifter` entry point
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Fully offline and deterministic (uses the hashing embedder; no GUI, network or keys).

## License

MIT.
