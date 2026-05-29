"""Configuration & tunable defaults for the Context Drift Monitor.

Everything is overridable via environment variables so the app, CLI, and tests
can point at different databases and thresholds without code changes.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- storage -----------------------------------------------------------------
DATA_DIR = Path(os.environ.get("CDM_DATA_DIR", str(Path.home() / ".context-drift-monitor")))
DB_PATH = Path(os.environ.get("CDM_DB_PATH", str(DATA_DIR / "cdm.db")))

# --- drift detection ---------------------------------------------------------
# Cosine-distance threshold above which a turn is flagged as "drifting". This
# default is calibrated for neural sentence embeddings (all-MiniLM-L6-v2), where
# on-topic turns sit well below it.
DEFAULT_THRESHOLD = float(os.environ.get("CDM_THRESHOLD", "0.65"))
# The pure-Python hashing fallback runs on a compressed, higher cosine-distance
# scale (even on-topic turns rarely drop below ~0.6), so it needs a higher
# threshold to separate on- from off-goal turns. Each embedder advertises its own
# suggested_threshold; this is the value the hashing fallback uses.
HASHING_THRESHOLD = float(os.environ.get("CDM_HASHING_THRESHOLD", "0.80"))
# Regenerate the goal-state snapshot every N user/assistant turns.
UPDATE_EVERY = int(os.environ.get("CDM_UPDATE_EVERY", "5"))
# Number of most-recent turns used for goal extraction and reference smoothing.
ROLLING_WINDOW = int(os.environ.get("CDM_WINDOW", "10"))
# Smooth the per-turn drift series with a trailing moving average of this size
# (1 = no smoothing). Lexical drift is noisy turn-to-turn — a small window lets
# the live graph read as a trend rather than jitter, while raw per-turn scores
# are still used for the immediate "this turn drifted" flag.
SMOOTHING_WINDOW = int(os.environ.get("CDM_SMOOTHING", "3"))

# --- embeddings --------------------------------------------------------------
# "auto"     -> sentence-transformers if importable, else the hashing fallback.
# "semantic" -> neural embeddings via fastembed/onnxruntime (no torch; downloads
#               the model once, then offline). "fastembed" is an alias.
# "local"    -> force sentence-transformers (errors if unavailable).
# "hashing"  -> force the pure-Python fallback embedder.
EMBEDDER_PREFERENCE = os.environ.get("CDM_EMBEDDER", "auto")
# Sentence-transformers model used when the local embedder is active.
LOCAL_MODEL_NAME = os.environ.get("CDM_LOCAL_MODEL", "all-MiniLM-L6-v2")
# fastembed model used by the semantic embedder (onnxruntime, no torch).
SEMANTIC_MODEL = os.environ.get("CDM_SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Where fastembed caches downloaded models (kept under the data dir, offline after).
SEMANTIC_CACHE = Path(os.environ.get("CDM_SEMANTIC_CACHE", str(DATA_DIR / "models")))
# Dimensionality of the pure-Python hashing embedder.
HASHING_DIM = int(os.environ.get("CDM_HASHING_DIM", "512"))


def ensure_data_dir() -> Path:
    """Create the data directory if needed and return it."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
