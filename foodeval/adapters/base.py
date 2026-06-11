"""Abstract protocol for embedding adapters.

All embedding adapters must satisfy the EmbeddingAdapter protocol. This enables
the benchmark runner to work uniformly with local models, API-based services,
and custom implementations.

Usage:
    >>> from foodeval.adapters.base import EmbeddingAdapter
    >>> def run_benchmark(adapter: EmbeddingAdapter, texts: list[str]):
    ...     embeddings = adapter.encode(texts)
    ...     print(f"{adapter.name}: {embeddings.shape[1]}d embeddings")
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

CACHE_DIR = Path("~/.cache/foodeval").expanduser()
CACHE_DISABLED = False


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """Protocol that all embedding adapters must satisfy."""

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: Input strings to encode.
            batch_size: Number of texts per forward pass or API call.
            normalize: Whether to L2-normalize the output vectors.

        Returns:
            Float32 array of shape (N, D) where N = len(texts) and
            D = self.dimension.
        """
        ...

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        ...

    @property
    def name(self) -> str:
        """Human-readable model name for leaderboard display."""
        ...


def encode_queries(
    adapter: EmbeddingAdapter,
    texts: list[str],
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """Encode query texts using an adapter's query role when available."""
    method = getattr(adapter, "encode_queries", None)
    if callable(method):
        return method(texts, batch_size=batch_size, normalize=normalize)
    return adapter.encode(texts, batch_size=batch_size, normalize=normalize)


def encode_documents(
    adapter: EmbeddingAdapter,
    texts: list[str],
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """Encode corpus/document texts using an adapter's document role when available."""
    method = getattr(adapter, "encode_documents", None)
    if callable(method):
        return method(texts, batch_size=batch_size, normalize=normalize)
    return adapter.encode(texts, batch_size=batch_size, normalize=normalize)


def _sanitize_adapter_name(adapter_name: str) -> str:
    """Sanitize adapter name for use in filesystem paths.

    Replaces path separators and traversal sequences with underscores so
    the name is safe to embed in a cache filename. This only sanitizes the
    name; callers are responsible for confirming the resolved path stays
    under CACHE_DIR (see load_cache and save_cache).
    """
    sanitized = re.sub(r"[/\\]|\.\.", "_", adapter_name)
    return sanitized


def cache_key(adapter_name: str, dimension: int, texts: list[str]) -> str:
    """Compute a deterministic cache key for a set of texts.

    The key is a SHA-256 hex digest of the adapter name, dimension, and
    text content in their original order. Cache only hits when texts
    arrive in the exact same order, since embeddings are stored positionally.

    Each text is hashed with an explicit byte length prefix and a NUL
    separator so the digest is injective: ``["foo\\nbar"]`` and
    ``["foo", "bar"]`` produce different keys.

    Args:
        adapter_name: Identifier for the model/adapter.
        dimension: Embedding dimension (affects output even for same model).
        texts: The texts being encoded.

    Returns:
        64-char hex digest string.
    """
    h = hashlib.sha256()
    h.update(str(len(adapter_name)).encode("ascii"))
    h.update(b"\x00")
    h.update(adapter_name.encode("utf-8"))
    h.update(str(dimension).encode("ascii"))
    h.update(b"\x00")
    h.update(str(len(texts)).encode("ascii"))
    h.update(b"\x00")
    for text in texts:
        encoded = text.encode("utf-8")
        h.update(str(len(encoded)).encode("ascii"))
        h.update(b"\x00")
        h.update(encoded)
    return h.hexdigest()


def load_cache(
    adapter_name: str, dimension: int, texts: list[str]
) -> np.ndarray | None:
    """Load cached embeddings if they exist.

    Args:
        adapter_name: Identifier for the model/adapter.
        dimension: Embedding dimension.
        texts: The texts that were encoded.

    Returns:
        Cached float32 array of shape (N, D), or None if not cached.
    """
    if CACHE_DISABLED:
        return None
    safe_name = _sanitize_adapter_name(adapter_name)
    key = cache_key(adapter_name, dimension, texts)
    path = (CACHE_DIR / f"{safe_name}_{key}.npy").resolve()
    if not str(path).startswith(str(CACHE_DIR.resolve())):
        return None
    if path.exists():
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 2 and arr.shape[0] == len(texts) and arr.shape[1] == dimension:
            return arr
    return None


def save_cache(
    adapter_name: str, dimension: int, texts: list[str], embeddings: np.ndarray
) -> None:
    """Save embeddings to disk cache.

    Args:
        adapter_name: Identifier for the model/adapter.
        dimension: Embedding dimension.
        texts: The texts that were encoded (used for cache key).
        embeddings: Float32 array of shape (N, D) to cache.
    """
    if CACHE_DISABLED:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_adapter_name(adapter_name)
    key = cache_key(adapter_name, dimension, texts)
    path = (CACHE_DIR / f"{safe_name}_{key}.npy").resolve()
    if not str(path).startswith(str(CACHE_DIR.resolve())):
        return
    np.save(path, embeddings.astype(np.float32))
