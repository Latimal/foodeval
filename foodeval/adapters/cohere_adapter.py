"""Cohere API adapter for embed-multilingual-v3 and embed-v4.

Uses the Cohere REST API directly (not via AWS Bedrock).
Requires a COHERE_API_KEY environment variable or trial key.

Usage:
    >>> from foodeval.adapters.cohere_adapter import CohereAdapter
    >>> adapter = CohereAdapter("embed-multilingual-v3.0", dimension=384)  # doctest: +SKIP
    >>> embeddings = adapter.encode(["butter chicken", "paneer tikka"])  # doctest: +SKIP
    >>> embeddings.shape  # doctest: +SKIP
    (2, 384)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import numpy as np

from foodeval.adapters.base import load_cache, save_cache

# Cohere API limits: 96 texts per request
_BATCH_SIZE = 96

# Per-request timeout (seconds) and bounded retry for transient errors.
_REQUEST_TIMEOUT = 30
_MAX_RETRIES = 3

# Known native dimensions
_NATIVE_DIMS = {
    "embed-multilingual-v3.0": 1024,
    "embed-english-v3.0": 1024,
    "embed-v4.0": 1536,
}


class CohereAdapter:
    """Adapter for Cohere Embed API (direct, not Bedrock).

    Args:
        model: Cohere model identifier (e.g. "embed-multilingual-v3.0").
        dimension: Output embedding dimension. Truncated and re-normalized
            for v3 models. v4 is fetched at native width and truncated client-side.
        api_key: Cohere API key. Falls back to COHERE_API_KEY env var.
    """

    def __init__(
        self,
        model: str,
        dimension: int | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._dimension = dimension
        self._api_key = api_key or os.environ.get("COHERE_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Cohere API key required. Set COHERE_API_KEY env var "
                "or pass api_key parameter."
            )
        self._is_v4 = "v4" in model

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts to embeddings via Cohere API.

        Args:
            texts: Input strings to encode.
            batch_size: Ignored (uses fixed 96-text Cohere batches).
            normalize: Whether to L2-normalize the output vectors.

        Returns:
            Float32 array of shape (N, D).
        """
        return self._encode_with_input_type(
            texts,
            input_type="search_document",
            normalize=normalize,
        )

    def encode_queries(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode query texts with Cohere's retrieval-query input type."""
        return self._encode_with_input_type(
            texts,
            input_type="search_query",
            normalize=normalize,
        )

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode corpus texts with Cohere's retrieval-document input type."""
        return self._encode_with_input_type(
            texts,
            input_type="search_document",
            normalize=normalize,
        )

    def _encode_with_input_type(
        self,
        texts: list[str],
        input_type: str,
        normalize: bool,
    ) -> np.ndarray:
        cached = load_cache(f"{self.name}:{input_type}", self.dimension, texts)
        if cached is not None:
            return cached

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            result = self._embed_batch(batch, input_type=input_type)

            # v2 API always nests under embedding_types
            embeddings = result["embeddings"]
            if isinstance(embeddings, dict):
                embeddings = embeddings.get("float", embeddings)
            all_embeddings.extend(embeddings)

            # Rate limiting between batches
            if i + _BATCH_SIZE < len(texts):
                time.sleep(0.5)

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        arr = np.array(all_embeddings, dtype=np.float32)

        # Truncate to requested dimension (v3 doesn't support native dim).
        # Truncation can only shrink: a requested dimension wider than the
        # native width is rejected rather than silently returned narrower.
        if self._dimension is not None:
            if arr.shape[1] < self._dimension:
                raise ValueError(
                    f"{self._model} returned {arr.shape[1]}d embeddings but "
                    f"dimension {self._dimension} was requested. Choose a "
                    f"value <= {arr.shape[1]}."
                )
            arr = arr[:, : self._dimension]

        if normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            arr = arr / norms

        arr = arr.astype(np.float32)
        if arr.shape[1] != self.dimension:
            raise ValueError(
                f"{self._model} produced {arr.shape[1]}d embeddings but "
                f"adapter reports dimension {self.dimension}."
            )
        save_cache(f"{self.name}:{input_type}", self.dimension, texts, arr)
        return arr

    def _embed_batch(self, batch: list[str], input_type: str) -> dict:
        """POST one batch to the Cohere embed endpoint with bounded retry.

        Retries transient network errors with exponential backoff. On an
        HTTP error, the response body (which carries Cohere's error message)
        is read and surfaced in the raised exception.
        """
        body = {
            "model": self._model,
            "texts": batch,
            "input_type": input_type,
            "embedding_types": ["float"],
            "truncate": "END",
        }
        data = json.dumps(body).encode()

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            req = urllib.request.Request(
                "https://api.cohere.com/v2/embed",
                data=data,
                method="POST",
            )
            req.add_header("Content-Type", "application/json")
            req.add_header("Accept", "application/json")
            req.add_header("Authorization", f"Bearer {self._api_key}")

            try:
                resp = urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT)
                return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")
                except Exception:  # noqa: BLE001 - body read is best-effort
                    pass
                last_exc = urllib.error.HTTPError(
                    exc.url,
                    exc.code,
                    f"{exc.reason}: {detail}" if detail else exc.reason,
                    exc.headers,
                    None,
                )
                # 4xx (except 429) are not transient; fail fast.
                if exc.code != 429 and 400 <= exc.code < 500:
                    raise last_exc from exc
                if attempt == _MAX_RETRIES - 1:
                    raise last_exc from exc
            except urllib.error.URLError as exc:
                last_exc = exc
                if attempt == _MAX_RETRIES - 1:
                    raise
            time.sleep(0.5 * (2**attempt))
        # Unreachable: the loop either returns or raises.
        raise last_exc  # type: ignore[misc]

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        if self._dimension is not None:
            return self._dimension
        return _NATIVE_DIMS.get(self._model, 1024)

    @property
    def name(self) -> str:
        """Human-readable model name for leaderboard display."""
        base = self._model.replace(".0", "")
        if self._dimension is not None:
            return f"{base}-{self._dimension}d"
        return base
