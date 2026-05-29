"""A tiny evaluation harness for drift detection.

Builds labeled synthetic conversations — each starts on a clear goal, stays on it
for a few turns, then drifts into unrelated topics at a known point — and checks
whether the changepoint detector fires *in the drifted region*. Reports
precision/recall so engine changes can be measured rather than eyeballed.

Run: ``drifter eval`` (or ``python -m cdm.eval``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from cdm.embeddings import get_embedder
from cdm.monitor import DriftMonitor
from cdm.storage import Store

# (goal, on-topic follow-ups, off-topic follow-ups)
CORPUS: List[Tuple[str, List[str], List[str]]] = [
    ("build a rust CLI that parses CSV files",
     ["add a flag to set the delimiter", "handle quoted fields with commas", "write unit tests for the parser"],
     ["what's a good banana bread recipe", "plan a weekend hiking trip", "which phone should I buy this year"]),
    ("design a pan-tilt camera gimbal under 5 kg",
     ["size the stepper motors for the tilt axis", "pick sealed bearings for the pan axis", "estimate the total mass budget"],
     ["recommend a netflix show for tonight", "how do I make cold brew coffee", "what's the weather like in tokyo"]),
    ("write a marketing email for our SaaS launch",
     ["draft a punchy subject line", "add a clear call to action", "keep the tone friendly and concise"],
     ["explain quantum entanglement", "best exercises for lower back pain", "history of the roman empire"]),
    ("debug a python memory leak in a web server",
     ["profile object allocations over time", "check for unclosed database connections", "look at the request handler lifecycle"],
     ["suggest a vegetarian dinner menu", "how to train for a marathon", "review of the latest superhero film"]),
    ("plan a 7-day itinerary for Japan",
     ["add two days in kyoto for temples", "include a day trip to nara", "budget for the bullet train pass"],
     ["fix my javascript build error", "what's a good resistance band workout", "explain how mortgages work"]),
]


def _session_turns(goal: str, on: List[str], off: List[str]) -> Tuple[List[Tuple[str, str]], int]:
    """Build (turns, drift_onset_turn_id) for one synthetic session."""
    turns: List[Tuple[str, str]] = [("assistant", f"Sure — let's work on: {goal}.")]
    for u in on:
        turns += [("user", u), ("assistant", f"Good point — {u}. Here's how...")]
    onset = len(turns)  # turn_id where off-topic begins (anchor is not a turn)
    for u in off:
        turns += [("user", u), ("assistant", f"Sure, about {u}...")]
    return turns, onset


def evaluate(embedder: str = "hashing", threshold: Optional[float] = None) -> Dict:
    """Run the corpus and return detection metrics.

    A session counts as a correct detection if a changepoint is flagged at or
    after (with a 1-turn grace) the labeled drift onset. Returns precision,
    recall, mean detection lead, and the raw counts.
    """
    emb = get_embedder(embedder)
    thr = threshold if threshold is not None else getattr(emb, "suggested_threshold", 0.8)
    detected = correct = 0
    total = len(CORPUS)
    for goal, on, off in CORPUS:
        mon = DriftMonitor(store=Store(":memory:"), embedder=emb, threshold=thr)
        s = mon.start_session("eval", goal, [])
        turns, onset = _session_turns(goal, on, off)
        mon.ingest_transcript(s.session_id, [{"role": r, "text": t} for r, t in turns])
        cp = mon.timeseries(s.session_id).get("changepoint_turn")
        if cp is not None:
            detected += 1
            if cp >= onset - 1:
                correct += 1
    precision = correct / detected if detected else 0.0
    recall = correct / total if total else 0.0
    return {
        "sessions": total, "detected": detected, "correct": correct,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "embedder": emb.name, "threshold": thr,
    }


def main(argv: Optional[list] = None) -> int:
    metrics = evaluate()
    print("Drift-detection eval on synthetic corpus:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
