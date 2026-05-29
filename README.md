# Drifter — Context Drift Monitor

Watch a long LLM conversation drift away from its original goal **in real time**, and
get a ready-to-paste corrective prompt to snap it back on course.

Over a long session an assistant slowly *forgets the goal* — it loosens early
constraints, accumulates small deviations, and ends up answering something adjacent to,
but no longer aligned with, what you actually wanted. Drifter sits beside the
conversation, **captures every message you copy automatically**, scores each turn's
drift, keeps a running estimate of your goal/constraints/decisions, and raises a
corrective prompt when drift crosses a threshold.

**Fully offline.** Clipboard access and scoring are local — no API keys, no network,
nothing leaves your machine.

---

## Install

```bash
pip install git+https://github.com/Divyonic/drifter.git
drifter
```

`drifter` launches the app in your browser. That's it.

For sharper *semantic* drift (instead of the offline lexical fallback), add the
optional neural embedder:

```bash
pip install "drifter[local] @ git+https://github.com/Divyonic/drifter.git"
```

### From a checkout

```bash
git clone https://github.com/Divyonic/drifter.git && cd drifter
./run.sh            # bootstraps a venv, installs deps, launches the app
```

---

## How to use it

1. **New session** (sidebar) — project name, your **initial goal** (the anchor), and any
   hard **constraints** (one per line). Or click **Load demo session** to see it work
   immediately on a sample conversation that drifts from gimbal engineering into snacks.
2. Leave Drifter open and **just chat with your LLM** (claude.ai, ChatGPT, a local
   model — anything). Every time you **copy** a prompt or a reply (Cmd/Ctrl+C), Drifter
   captures it as a turn. The drift graph **updates itself live** — no clicking.
3. When a turn crosses the threshold you get a **warning** and a **corrective prompt**.
   Copy it, paste it into your chat, then hit **Mark checkpoint** to re-anchor.

Sessions persist and **auto-resume** — reopen Drifter and it picks up where you left off.

You can also feed turns manually (**Add turn**) or bulk-import a transcript
(JSON or `User:`/`Assistant:` markdown).

### Commands

```bash
drifter          # launch the web app (default)
drifter watch    # run the clipboard watcher on its own (the app starts it for you)
drifter version
```

---

## How it works

```
you copy a turn ─▶ clipboard watcher ─▶ embed ─▶ drift score ─┬─▶ live graph (vs anchor & rolling reference)
                                            │                  └─▶ threshold crossed? ─▶ corrective prompt
                                            └─▶ every N turns: re-derive goal state (goal · constraints · decisions · focus)
```

- **Auto-capture.** A background watcher (`cdm/watcher.py`) polls the clipboard and
  appends new copies to the active session. It alternates roles from history, skips
  duplicates, ignores tiny snippets, and never re-captures the corrective prompt itself.
- **Anchor vs reference.** `drift_from_anchor` is the cosine distance from a turn to your
  *original* goal (never changes); `drift_from_reference` is the distance to the current
  goal-state snapshot, re-derived every few turns. A turn is flagged when either exceeds
  the threshold.
- **Goal-state extraction is heuristic and LLM-free.** Constraints are mined from your
  messages (strong modals like *must / required / non-negotiable*, plus numeric limits
  like `< 5 kg`, `10 L`, `$200`); decisions from choice phrases (*chose, decided, go
  with, lock in…*); `current_focus` from the salient keywords of recent turns. Your
  original goal is preserved verbatim as `core_goal`.
- **Smoothing.** Per-turn distance is noisy, so the graph shows a short trailing moving
  average — the *trend* is the signal. The raw per-turn score still drives the
  immediate "this turn drifted" flag.

### Embeddings & calibration

| Backend | When | Notes |
|---|---|---|
| **HashingEmbedder** (default) | always available | Pure-Python + numpy. Deterministic, offline, zero setup. Measures **lexical/topical** drift (word + char n-gram overlap). |
| **LocalEmbedder** (`all-MiniLM-L6-v2`) | if `sentence-transformers` is installed | Neural **semantic** embeddings — separates "related but reworded" from "off-topic" far better. |

The hashing fallback runs on a compressed, higher distance scale, so each backend
advertises its own threshold (0.80 hashing / 0.65 neural) and the app defaults the
slider accordingly. It's excellent at catching drift that involves a **change of
vocabulary** (the usual case — the demo climbs cleanly as the topic shifts). Short,
on-topic turns against a long anchor can read high under the lexical fallback; install
`sentence-transformers` for semantic accuracy (auto-detected on next launch).

---

## Configuration

All tunables are environment variables (defaults shown):

| Var | Default | Meaning |
|---|---|---|
| `CDM_DB_PATH` | `~/.context-drift-monitor/cdm.db` | SQLite database location |
| `CDM_THRESHOLD` | `0.65` | Drift threshold for neural embeddings |
| `CDM_HASHING_THRESHOLD` | `0.80` | Drift threshold for the hashing fallback |
| `CDM_UPDATE_EVERY` | `5` | Re-derive the goal state every N turns |
| `CDM_WINDOW` | `10` | Recent-turn window for goal extraction |
| `CDM_SMOOTHING` | `3` | Trailing moving-average window for the graph |
| `CDM_EMBEDDER` | `auto` | `auto` / `local` / `hashing` |

---

## Use it as a library

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
  cli.py          the `drifter` command
  app.py          Streamlit UI (live, self-refreshing)
tests/            pytest suite (offline, deterministic)
sample_transcript.json   14-turn demo that starts on-goal and drifts
pyproject.toml    packaging + `drifter` entry point
run.sh            venv bootstrap + launch (from a checkout)
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Fully offline and deterministic (uses the hashing embedder).

## License

MIT.
