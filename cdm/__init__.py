"""Context Drift Monitor — track how far an LLM conversation has drifted from its
original goal, and generate a ready-to-paste corrective prompt to re-align it.

Public surface:
    from cdm.models import Session, Message, GoalState, DriftScore
    from cdm.embeddings import get_embedder, cosine_similarity, cosine_distance
    from cdm.drift import DriftEngine
    from cdm.goal_state import extract_goal_state, goal_state_text
    from cdm.corrective import render_corrective_prompt, CORRECTIVE_TEMPLATE
    from cdm.transcript import parse_transcript
    from cdm.storage import Store
    from cdm.monitor import DriftMonitor
"""

__version__ = "0.1.0"
