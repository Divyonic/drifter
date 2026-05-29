"""Tests for cdm.embeddings.

These run fully offline using the pure-Python HashingEmbedder; no network, no API
keys, and no sentence-transformers required.
"""

from __future__ import annotations

import math

import pytest

from cdm import config
from cdm.embeddings import (
    Embedder,
    HashingEmbedder,
    cosine_distance,
    cosine_similarity,
    get_embedder,
)


# --- HashingEmbedder basics --------------------------------------------------


def test_hashing_embedder_dim_and_name() -> None:
    emb = HashingEmbedder(dim=256)
    assert emb.dim == 256
    assert emb.name == "hashing:256"


def test_hashing_embedder_default_dim_from_config() -> None:
    emb = HashingEmbedder()
    assert emb.dim == config.HASHING_DIM


def test_hashing_embedder_rejects_nonpositive_dim() -> None:
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)


def test_satisfies_embedder_protocol() -> None:
    emb = HashingEmbedder()
    assert isinstance(emb, Embedder)


def test_encode_returns_correct_shape() -> None:
    emb = HashingEmbedder(dim=128)
    vecs = emb.encode(["hello world", "a second sentence"])
    assert len(vecs) == 2
    assert all(len(v) == 128 for v in vecs)
    assert all(isinstance(x, float) for v in vecs for x in v)


def test_encode_one_matches_batch() -> None:
    emb = HashingEmbedder(dim=128)
    text = "design a lightweight gimbal mount"
    one = emb.encode_one(text)
    batch = emb.encode([text])[0]
    assert one == batch


# --- determinism -------------------------------------------------------------


def test_hashing_embedder_deterministic_same_instance() -> None:
    emb = HashingEmbedder(dim=200)
    text = "deterministic embeddings are important"
    assert emb.encode_one(text) == emb.encode_one(text)


def test_hashing_embedder_deterministic_across_instances() -> None:
    text = "the quick brown fox jumps over the lazy dog"
    a = HashingEmbedder(dim=256).encode_one(text)
    b = HashingEmbedder(dim=256).encode_one(text)
    assert a == b


# --- unit-normalisation ------------------------------------------------------


def test_nonempty_vector_is_unit_norm() -> None:
    emb = HashingEmbedder()
    vec = emb.encode_one("some meaningful content here")
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm == pytest.approx(1.0, abs=1e-9)


def test_empty_text_yields_zero_vector() -> None:
    emb = HashingEmbedder(dim=64)
    for text in ["", "   ", "\n\t "]:
        vec = emb.encode_one(text)
        assert len(vec) == 64
        assert all(x == 0.0 for x in vec)


# --- semantic ordering: related > unrelated ----------------------------------


def test_paraphrase_more_similar_than_unrelated() -> None:
    emb = HashingEmbedder()
    base = "design a pan-tilt camera mount that weighs under five kilograms"
    paraphrase = "build a lightweight pan and tilt camera mounting under 5 kg"
    unrelated = "the office ran out of coffee and snacks this morning"

    related_sim = cosine_similarity(
        emb.encode_one(base), emb.encode_one(paraphrase)
    )
    unrelated_sim = cosine_similarity(
        emb.encode_one(base), emb.encode_one(unrelated)
    )
    assert related_sim > unrelated_sim


def test_topically_related_more_similar_than_unrelated() -> None:
    emb = HashingEmbedder()
    base = "machine learning models for radar signal classification"
    related = "training a neural network to classify radar signals"
    unrelated = "best recipes for a vegetarian lasagna dinner"

    rel = cosine_similarity(emb.encode_one(base), emb.encode_one(related))
    unrel = cosine_similarity(emb.encode_one(base), emb.encode_one(unrelated))
    assert rel > unrel


def test_identical_text_has_similarity_one() -> None:
    emb = HashingEmbedder()
    v = emb.encode_one("exactly the same string of words")
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-9)


# --- cosine_similarity / cosine_distance edge cases --------------------------


def test_cosine_similarity_in_range() -> None:
    emb = HashingEmbedder()
    a = emb.encode_one("alpha beta gamma delta")
    b = emb.encode_one("completely different epsilon zeta")
    sim = cosine_similarity(a, b)
    assert -1.0 <= sim <= 1.0


def test_cosine_similarity_empty_vectors_zero() -> None:
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0, 2.0], []) == 0.0
    assert cosine_similarity([], [1.0, 2.0]) == 0.0


def test_cosine_similarity_zero_vector_zero() -> None:
    assert cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0
    assert cosine_similarity([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == 0.0


def test_cosine_similarity_length_mismatch_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_cosine_distance_in_range() -> None:
    emb = HashingEmbedder()
    samples = [
        "design a lightweight gimbal",
        "office coffee machine broke",
        "neural network radar classifier",
        "vegetarian lasagna recipe ideas",
    ]
    vecs = emb.encode(samples)
    for i in range(len(vecs)):
        for j in range(len(vecs)):
            d = cosine_distance(vecs[i], vecs[j])
            assert 0.0 <= d <= 2.0


def test_cosine_distance_identical_is_zero() -> None:
    emb = HashingEmbedder()
    v = emb.encode_one("identical content here")
    assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-9)


def test_cosine_distance_empty_is_one() -> None:
    assert cosine_distance([], []) == 1.0
    assert cosine_distance([0.0, 0.0], [1.0, 1.0]) == 1.0


def test_cosine_distance_opposite_vectors_two() -> None:
    assert cosine_distance([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0, abs=1e-9)


def test_cosine_distance_matches_similarity() -> None:
    emb = HashingEmbedder()
    a = emb.encode_one("first body of text about mounts")
    b = emb.encode_one("second body of text about mounts and gimbals")
    assert cosine_distance(a, b) == pytest.approx(
        1.0 - cosine_similarity(a, b), abs=1e-12
    )


# --- get_embedder factory ----------------------------------------------------


def test_get_embedder_hashing() -> None:
    emb = get_embedder("hashing")
    assert isinstance(emb, HashingEmbedder)
    assert emb.name.startswith("hashing:")


def test_get_embedder_auto_returns_embedder() -> None:
    # Offline this falls back to HashingEmbedder; either way it must satisfy
    # the protocol and produce unit-normalised vectors.
    emb = get_embedder("auto")
    assert isinstance(emb, Embedder)
    vec = emb.encode_one("a probe sentence")
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_get_embedder_unknown_preference_raises() -> None:
    with pytest.raises(ValueError):
        get_embedder("nonsense")


def test_get_embedder_case_insensitive() -> None:
    assert isinstance(get_embedder("HASHING"), HashingEmbedder)
