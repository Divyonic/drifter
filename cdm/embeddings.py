"""Embedding backends for the Context Drift Monitor.

Two interchangeable embedders implement the :class:`Embedder` protocol:

* :class:`HashingEmbedder` — a pure-Python + numpy, deterministic, network-free
  fallback. It hashes word unigrams and character 3-5 grams into signed buckets,
  applies sublinear term-frequency weighting, and L2-normalises the result. It is
  always available and gives higher cosine similarity to paraphrases / topically
  related text than to unrelated text.
* :class:`LocalEmbedder` — wraps ``sentence-transformers`` (lazily imported). If the
  package or model is unavailable it raises so the factory can fall back.

The core package only depends on the standard library and numpy;
``sentence-transformers`` is optional and may be absent.
"""

from __future__ import annotations

import hashlib
import math
from typing import List, Protocol, runtime_checkable

import numpy as np

from cdm import config

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "LocalEmbedder",
    "SemanticEmbedder",
    "get_embedder",
    "cosine_similarity",
    "cosine_distance",
]


@runtime_checkable
class Embedder(Protocol):
    """Protocol implemented by every embedding backend.

    Implementations expose a human-readable ``name`` (e.g. ``"hashing:512"`` or
    ``"local:all-MiniLM-L6-v2"``), the embedding ``dim``, a ``suggested_threshold``
    (the cosine-distance value at which drift is best flagged for this backend's
    distance scale), and batch / single encoding methods that return
    **unit-normalised** vectors as ``list[float]``.
    """

    name: str
    dim: int
    suggested_threshold: float

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts into unit-normalised vectors."""
        ...

    def encode_one(self, text: str) -> List[float]:
        """Encode a single text into a unit-normalised vector."""
        ...


def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` scaled to unit L2 norm; an all-zero vector is left as-is."""
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


class HashingEmbedder:
    """Pure-Python + numpy, deterministic, zero-dependency fallback embedder.

    Hashes three kinds of feature into ``dim`` buckets using a stable SHA-1-based
    hash: word unigrams, word bigrams, and character 3-5 grams. Word features
    carry the topical signal that matters for drift, so they are weighted well
    above the character n-grams (which add robustness to morphology/typos but,
    left unchecked, swamp the word signal and compress every distance toward 1).
    Each feature contributes a signed count (the sign is derived from a second
    hash bit) so distinct features can partially cancel, reducing collision bias;
    counts are sublinearly weighted with ``1 + log(count)`` per feature kind, the
    kinds are summed with their weights, and the vector is L2-normalised.

    The embedder is fully deterministic: identical input text always yields an
    identical vector, independent of process or platform.
    """

    # Relative weights per feature kind. Word overlap dominates (topical signal);
    # character n-grams contribute a smaller robustness term.
    _WEIGHT_UNIGRAM = 1.0
    _WEIGHT_BIGRAM = 0.7
    _WEIGHT_CHAR = 0.25

    def __init__(self, dim: int = config.HASHING_DIM) -> None:
        """Create a hashing embedder writing into ``dim`` buckets.

        Args:
            dim: Number of output dimensions / hash buckets. Must be positive.
        """
        if dim <= 0:
            raise ValueError("HashingEmbedder dim must be a positive integer")
        self.dim: int = int(dim)
        self.name: str = f"hashing:{self.dim}"
        # The hashing scale runs high even for on-topic text; flag drift higher.
        self.suggested_threshold: float = config.HASHING_THRESHOLD

    @staticmethod
    def _tokens(text: str) -> List[str]:
        """Split text into lowercase word-unigram tokens (alphanumeric runs)."""
        tokens: List[str] = []
        current: List[str] = []
        for ch in text.lower():
            if ch.isalnum():
                current.append(ch)
            elif current:
                tokens.append("".join(current))
                current = []
        if current:
            tokens.append("".join(current))
        return tokens

    @staticmethod
    def _char_ngrams(text: str) -> List[str]:
        """Yield character 3-, 4- and 5-grams over a whitespace-collapsed string."""
        collapsed = " ".join(text.lower().split())
        if not collapsed:
            return []
        ngrams: List[str] = []
        length = len(collapsed)
        for n in (3, 4, 5):
            if length < n:
                continue
            for i in range(length - n + 1):
                ngrams.append(collapsed[i : i + n])
        return ngrams

    def _word_bigrams(self, text: str) -> List[str]:
        """Yield consecutive word-bigram strings (e.g. ``"tilt motor"``)."""
        toks = self._tokens(text)
        return [f"{toks[i]} {toks[i + 1]}" for i in range(len(toks) - 1)]

    def _feature_groups(self, text: str) -> List[tuple]:
        """Return ``(namespaced_features, weight)`` pairs, one per feature kind."""
        return [
            ([f"w\x00{t}" for t in self._tokens(text)], self._WEIGHT_UNIGRAM),
            ([f"b\x00{g}" for g in self._word_bigrams(text)], self._WEIGHT_BIGRAM),
            ([f"c\x00{g}" for g in self._char_ngrams(text)], self._WEIGHT_CHAR),
        ]

    @staticmethod
    def _hash(feature: str) -> int:
        """Return a stable non-negative integer hash for a feature string."""
        digest = hashlib.sha1(feature.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def _accumulate(self, feats: List[str]) -> np.ndarray:
        """Signed, sublinear-tf vector for a single feature kind (pre-weighting)."""
        out = np.zeros(self.dim, dtype=np.float64)
        if not feats:
            return out
        counts = np.zeros(self.dim, dtype=np.float64)
        for feat in feats:
            h = self._hash(feat)
            bucket = h % self.dim
            sign = 1.0 if (h // self.dim) & 1 else -1.0
            counts[bucket] += sign
        # Sublinear tf: 1 + log(|count|) preserving sign; zero stays zero.
        nz = counts != 0.0
        out[nz] = np.sign(counts[nz]) * (1.0 + np.log(np.abs(counts[nz])))
        return out

    def _vector(self, text: str) -> np.ndarray:
        """Build the raw (pre-normalisation) weighted signed sublinear-tf vector.

        Each feature kind (word unigrams, word bigrams, char n-grams) is reduced to
        its own signed sublinear-tf vector, then the kinds are summed with their
        relative weights so word overlap dominates the topical signal.
        """
        vec = np.zeros(self.dim, dtype=np.float64)
        if not text or not text.strip():
            return vec
        for feats, weight in self._feature_groups(text):
            if feats:
                vec += weight * self._accumulate(feats)
        return vec

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts into unit-normalised vectors.

        Args:
            texts: Input strings; empty / whitespace strings yield a zero vector.

        Returns:
            A list of ``dim``-length lists of floats, one per input text.
        """
        out: List[List[float]] = []
        for text in texts:
            vec = _l2_normalise(self._vector(text))
            out.append(vec.tolist())
        return out

    def encode_one(self, text: str) -> List[float]:
        """Encode a single text into a unit-normalised vector."""
        return self.encode([text])[0]


class LocalEmbedder:
    """Embedder backed by ``sentence-transformers`` (lazily imported).

    The model is loaded eagerly in ``__init__`` so that an unavailable package or
    model fails fast with a helpful error that :func:`get_embedder` can catch and
    fall back from.
    """

    def __init__(self, model_name: str = config.LOCAL_MODEL_NAME) -> None:
        """Load ``model_name`` via sentence-transformers.

        Args:
            model_name: Name or path of the sentence-transformers model.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
            RuntimeError: If the model cannot be loaded.
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised only when absent
            raise ImportError(
                "LocalEmbedder requires the optional 'sentence-transformers' "
                "package. Install it (pip install sentence-transformers) or use "
                "the hashing embedder."
            ) from exc

        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # pragma: no cover - needs network/model files
            raise RuntimeError(
                f"Could not load sentence-transformers model {model_name!r}: {exc}. "
                "Falling back to the hashing embedder is recommended."
            ) from exc

        self.model_name: str = model_name
        self.name: str = f"local:{model_name}"
        # Neural embeddings separate on- from off-topic well at the default scale.
        self.suggested_threshold: float = config.DEFAULT_THRESHOLD
        try:
            dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:  # pragma: no cover - depends on model internals
            dim = len(self._encode_raw([""])[0])
        self.dim: int = dim

    def _encode_raw(self, texts: List[str]):
        """Run the underlying model and return numpy vectors."""
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts into unit-normalised vectors."""
        if not texts:
            return []
        raw = np.asarray(self._encode_raw(texts), dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        return [_l2_normalise(row).tolist() for row in raw]

    def encode_one(self, text: str) -> List[float]:
        """Encode a single text into a unit-normalised vector."""
        return self.encode([text])[0]


class SemanticEmbedder:
    """Neural sentence embeddings via ``fastembed`` (onnxruntime — no torch).

    Downloads the model once (cached under the data dir), then runs fully offline.
    Gives proper *semantic* drift: "related but reworded" text scores close, while
    off-topic text scores far — unlike the lexical hashing fallback. ``fastembed``
    is lazily imported so the package works without it.
    """

    def __init__(self, model_name: str = config.SEMANTIC_MODEL) -> None:
        try:
            from fastembed import TextEmbedding  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised only when absent
            raise ImportError(
                "SemanticEmbedder requires the optional 'fastembed' package "
                "(pip install fastembed)."
            ) from exc
        try:
            config.SEMANTIC_CACHE.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(
                model_name=model_name, cache_dir=str(config.SEMANTIC_CACHE)
            )
        except Exception as exc:  # pragma: no cover - needs network/model files
            raise RuntimeError(
                f"Could not load fastembed model {model_name!r}: {exc}"
            ) from exc
        self.model_name: str = model_name
        self.name: str = f"semantic:{model_name.split('/')[-1]}"
        # Neural embeddings separate on- from off-topic at the default scale.
        self.suggested_threshold: float = config.DEFAULT_THRESHOLD
        self.dim: int = len(self.encode_one("probe"))

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts into unit-normalised vectors."""
        if not texts:
            return []
        out: List[List[float]] = []
        for vec in self._model.embed(list(texts)):
            out.append(_l2_normalise(np.asarray(vec, dtype=np.float64)).tolist())
        return out

    def encode_one(self, text: str) -> List[float]:
        """Encode a single text into a unit-normalised vector."""
        return self.encode([text])[0]


def get_embedder(preference: str = config.EMBEDDER_PREFERENCE) -> Embedder:
    """Build an :class:`Embedder` according to ``preference``.

    Args:
        preference: One of ``"auto"``, ``"local"`` or ``"hashing"``.

            * ``"auto"`` — try :class:`LocalEmbedder`, fall back to
              :class:`HashingEmbedder` on any error.
            * ``"local"`` — force :class:`LocalEmbedder` (may raise).
            * ``"hashing"`` — force :class:`HashingEmbedder`.

    Returns:
        An embedder instance satisfying the :class:`Embedder` protocol.

    Raises:
        ValueError: If ``preference`` is not a recognised value.
    """
    pref = (preference or "auto").strip().lower()
    if pref == "hashing":
        return HashingEmbedder()
    if pref in ("semantic", "fastembed"):
        return SemanticEmbedder()
    if pref == "local":
        return LocalEmbedder()
    if pref == "auto":
        try:
            return LocalEmbedder()
        except Exception:
            return HashingEmbedder()
    raise ValueError(
        f"Unknown embedder preference {preference!r}; "
        "expected 'auto', 'semantic', 'local' or 'hashing'."
    )


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in ``[-1, 1]``. Returns ``0.0`` if either vector is
        empty, all-zero, or the two differ in length.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    sim = float(np.dot(va, vb) / (na * nb))
    # Guard against tiny floating-point excursions outside [-1, 1].
    if sim > 1.0:
        return 1.0
    if sim < -1.0:
        return -1.0
    return sim


def cosine_distance(a: List[float], b: List[float]) -> float:
    """Cosine distance ``1 - cosine_similarity`` clamped to ``[0, 2]``.

    Empty / zero vectors produce a similarity of ``0.0`` and therefore a
    distance of ``1.0``.
    """
    dist = 1.0 - cosine_similarity(a, b)
    if dist < 0.0:
        return 0.0
    if dist > 2.0:
        return 2.0
    return dist
