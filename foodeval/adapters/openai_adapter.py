"""OpenAI API adapter for text-embedding models.

Supports OpenAI's embedding models with native Matryoshka dimension
parameter and disk caching.

Usage:
    >>> from foodeval.adapters.openai_adapter import OpenAIAdapter
    >>> adapter = OpenAIAdapter("text-embedding-3-large", dimension=384)  # doctest: +SKIP
    >>> embeddings = adapter.encode(["butter chicken", "paneer tikka"])  # doctest: +SKIP
    >>> embeddings.shape  # doctest: +SKIP
    (2, 384)
"""

from __future__ import annotations

import os
import time

import numpy as np

from foodeval.adapters.base import load_cache, save_cache

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]

# OpenAI embedding API accepts up to 2048 texts, but we cap lower
# to keep request sizes reasonable.
_MAX_BATCH_SIZE = 100

# Per-request timeout (seconds) and bounded retry for transient errors.
_REQUEST_TIMEOUT = 60.0
_MAX_RETRIES = 3


class OpenAIAdapter:
    """Adapter for OpenAI embedding models.

    Args:
        model: OpenAI model ID (e.g. "text-embedding-3-large",
            "text-embedding-3-small").
        dimension: Output embedding dimension. Passed as the
            ``dimensions`` parameter in the API request for
            Matryoshka-style truncation.
        api_key: OpenAI API key. If None, reads from the
            OPENAI_API_KEY environment variable.
        base_url: Custom API base URL. If None, reads from
            OPENAI_BASE_URL env var, then falls back to OpenAI default.
            Use for GitHub Models, Azure, or compatible endpoints.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        dimension: int = 384,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if openai is None:
            raise ImportError(
                "openai is required for OpenAIAdapter. "
                "Install it with: pip install 'foodeval[api]'"
            )

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No API key provided. Pass api_key= or set OPENAI_API_KEY."
            )

        resolved_base = base_url or os.environ.get("OPENAI_BASE_URL")
        kwargs: dict = {"api_key": resolved_key, "timeout": _REQUEST_TIMEOUT}
        if resolved_base:
            kwargs["base_url"] = resolved_base

        self._client = openai.OpenAI(**kwargs)
        self._model = model
        self._dimension = dimension

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts to embeddings via OpenAI API.

        Args:
            texts: Input strings to encode.
            batch_size: Number of texts per API call. Capped at 100.
            normalize: Whether to L2-normalize the output vectors.
                OpenAI returns normalized vectors by default, but this
                ensures consistency after any truncation.

        Returns:
            Float32 array of shape (N, D).
        """
        cached = load_cache(self.name, self.dimension, texts)
        if cached is not None:
            return cached

        effective_batch = min(batch_size, _MAX_BATCH_SIZE)
        all_embeddings: list[list[float]] = [[] for _ in range(len(texts))]

        for i in range(0, len(texts), effective_batch):
            batch = texts[i : i + effective_batch]
            response = self._embed_batch(batch)
            # API returns results in order but we use the index to be safe
            for item in response.data:
                all_embeddings[i + item.index] = item.embedding

        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        embeddings = np.array(all_embeddings, dtype=np.float32)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            embeddings = embeddings / norms

        if embeddings.shape[1] != self._dimension:
            raise ValueError(
                f"{self._model} returned {embeddings.shape[1]}d embeddings "
                f"but adapter reports dimension {self._dimension}. The model "
                "may not support the requested 'dimensions' value."
            )

        save_cache(self.name, self.dimension, texts, embeddings)
        return embeddings

    def _embed_batch(self, batch: list[str]):
        """Call the embeddings API with a bounded retry and backoff.

        The OpenAI client applies its own retries; this adds a backstop for
        transient errors. The final attempt's exception propagates unchanged.
        """
        for attempt in range(_MAX_RETRIES):
            try:
                return self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                    dimensions=self._dimension,
                )
            except Exception:  # noqa: BLE001 - re-raised after retries
                if attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(0.5 * (2**attempt))

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self._dimension

    @property
    def name(self) -> str:
        """Human-readable model name for leaderboard display."""
        return f"{self._model}-{self._dimension}d"
