"""Lexical baseline adapter (implements EmbeddingAdapter protocol).

Produces fixed-dimension sparse-to-dense vectors via hashed term frequency
so the standard evaluation harness (cosine similarity, NDCG, F1 threshold
sweep, linear probe) works without task-level changes. This is a legitimate
lexical baseline: it tests whether simple term overlap solves the task.

Despite the class name ``BM25Adapter`` and CLI key ``bm25`` (kept for
backward compatibility), this is NOT Okapi BM25. It is a hashed term-frequency
vectorizer with no IDF weighting and no sublinear TF scaling. Term counts are
L2-normalized so cosine similarity reduces to normalized term overlap.

Usage:
    >>> from foodeval.adapters.bm25 import BM25Adapter
    >>> adapter = BM25Adapter()
    >>> embeddings = adapter.encode(["butter chicken", "iced tea"])
    >>> embeddings.shape
    (2, 4096)
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


# Fixed dimension for the hashed term-frequency vectors. 4096 is large
# enough to avoid excessive hash collisions on food domain vocabulary while
# staying small enough for fast cosine computation.
_DEFAULT_DIM = 4096


class BM25Adapter:
    """Lexical TF baseline using hashed term-frequency vectors.

    Satisfies the EmbeddingAdapter protocol so it flows through the
    standard evaluation pipeline. The underlying representation is a
    bag-of-words term-frequency vector (no IDF, no sublinear scaling),
    projected to a fixed dimension via the hashing trick and L2-normalized.
    Named ``BM25Adapter`` for backward compatibility only; it is not
    Okapi BM25.
    """

    def __init__(self, dim: int = _DEFAULT_DIM) -> None:
        self._dim = dim
        self._vectorizer = HashingVectorizer(
            n_features=dim,
            alternate_sign=False,  # all positive weights
            norm="l2",
            analyzer="word",
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
        )

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts to hashed term-frequency vectors.

        Args:
            texts: Input strings to encode.
            batch_size: Ignored (vectorization is already fast).
            normalize: Whether to L2-normalize. HashingVectorizer already
                normalizes by default, but we re-normalize dense output
                for safety.

        Returns:
            Float32 array of shape (N, dim).
        """
        sparse = self._vectorizer.transform(texts)
        dense = np.asarray(sparse.todense(), dtype=np.float32)

        if normalize:
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            dense = dense / norms

        return dense

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self._dim

    @property
    def name(self) -> str:
        """Human-readable model name for leaderboard display."""
        return "Lexical (TF)"
