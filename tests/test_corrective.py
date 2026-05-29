"""Tests for cdm.corrective — corrective-prompt rendering."""

from __future__ import annotations

from cdm.corrective import CORRECTIVE_TEMPLATE, render_corrective_prompt


def test_all_slots_filled() -> None:
    """A fully-populated goal state fills every slot with bullet lists."""
    raw = {
        "core_goal": "design a pan-tilt EO/IR mount under 5 kg",
        "constraints": ["mass < 5 kg", "must use brushless motors"],
        "decisions": ["chose carbon fibre frame", "selected harmonic drives"],
        "current_focus": "gimbal bearing selection",
    }
    out = render_corrective_prompt(raw)

    assert "Core goal: design a pan-tilt EO/IR mount under 5 kg" in out
    assert "- mass < 5 kg" in out
    assert "- must use brushless motors" in out
    assert "- chose carbon fibre frame" in out
    assert "- selected harmonic drives" in out
    assert "Current focus: gimbal bearing selection" in out
    assert "(none specified)" not in out


def test_empty_lists_render_none_specified() -> None:
    """Empty constraint/decision lists collapse to '(none specified)'."""
    raw = {
        "core_goal": "ship the MVP",
        "constraints": [],
        "decisions": [],
        "current_focus": "writing tests",
    }
    out = render_corrective_prompt(raw)

    assert out.count("(none specified)") == 2
    # The constraint/decision slots themselves collapse to "(none specified)"
    # rather than emitting any data bullets. (The template's own fixed
    # instruction lines legitimately start with "- ", so a blanket
    # `"- " not in out` check would be impossible — assert the slots instead.)
    assert "  - Constraints:\n(none specified)" in out
    assert "  - Recent decisions:\n(none specified)" in out


def test_output_contains_core_goal_text() -> None:
    """The core_goal text appears verbatim in the rendered prompt."""
    goal = "build a counter-UAV adversarial radar dataset"
    out = render_corrective_prompt({"core_goal": goal})
    assert goal in out


def test_missing_keys_tolerated() -> None:
    """A dict missing every key still renders without raising."""
    out = render_corrective_prompt({})
    assert "Core goal:" in out
    assert "Current focus:" in out
    assert out.count("(none specified)") == 2


def test_empty_dict_and_none_safe() -> None:
    """An empty or None argument is handled gracefully."""
    assert isinstance(render_corrective_prompt({}), str)
    assert isinstance(render_corrective_prompt(None), str)  # type: ignore[arg-type]


def test_blank_entries_skipped() -> None:
    """Whitespace-only list entries are dropped, not rendered as bullets."""
    raw = {
        "core_goal": "x",
        "constraints": ["  ", "real constraint", ""],
        "decisions": ["", "   "],
        "current_focus": "y",
    }
    out = render_corrective_prompt(raw)
    assert "- real constraint" in out
    # decisions were all blank -> none specified; constraints had one real item
    assert out.count("(none specified)") == 1


def test_uses_template_structure() -> None:
    """Rendered output preserves the template's fixed instruction lines."""
    out = render_corrective_prompt({"core_goal": "g", "current_focus": "f"})
    assert out.startswith("You are acting as a strict alignment assistant.")
    assert "- If any future message of mine contradicts these" in out
    assert out.rstrip().endswith("then continue.")
    # template has the expected slot scaffolding
    assert "{core_goal}" in CORRECTIVE_TEMPLATE
    assert "{constraints}" in CORRECTIVE_TEMPLATE
    assert "{decisions}" in CORRECTIVE_TEMPLATE
    assert "{current_focus}" in CORRECTIVE_TEMPLATE
