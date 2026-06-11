"""Voyage AI embedding adapter (via the MongoDB-hosted Voyage endpoint).

Uses the Voyage embeddings REST API directly. Requires a VOYAGE_API_KEY
environment variable (or an api_key argument). The base URL defaults to the
MongoDB-hosted Voyage endpoint and can be overridden via VOYAGE_BASE_URL.

Voyage supports native output dimensions {256, 512, 1024, 2048}. To match the
parity treatment applied to the other 384d models, this adapter requests the
smallest native width that is >= the requested dimension, then truncates to
the requested dimension and L2-renormalizes.

Usage:
    >>> from foodeval.adapters.voyage_adapter import VoyageAdapter
    >>> adapter = VoyageAdapter("voyage-4-large", dimension=384)  # doctest: +SKIP
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

# Voyage accepts up to 128 inputs per embeddings request.
_BATCH_SIZE = 128

# Per-request timeout (seconds) and bounded retry for transient errors.
_REQUEST_TIMEOUT = 30
_MAX_RETRIES = 8

# Minimum seconds between requests. The MongoDB-hosted free tier caps at
# 3 RPM, so ~21s spacing keeps us under the limit. Override with the
# VOYAGE_MIN_INTERVAL env var (set to 0 for paid tiers with higher limits).
_DEFAULT_MIN_INTERVAL = 21.0

# Backoff (seconds) applied after a 429 before the next retry. The free-tier
# window is one minute, so a long pause clears the rate-limit counter.
_RATE_LIMIT_BACKOFF = 21.0

# Native output dimensions Voyage can return directly.
_NATIVE_DIMS = (256, 512, 1024, 2048)

# Default endpoint: the MongoDB-hosted Voyage embeddings service.
_DEFAULT_BASE_URL = "https://ai.mongodb.com/v1"


def _native_output_dimension(requested: int) -> int:
    """Return the smallest supported native width that is >= requested.

    Falls back to the largest native width if the requested dimension exceeds
    every supported value.
    """
    for native in _NATIVE_DIMS:
        if native >= requested:
            return native
    return _NATIVE_DIMS[-1]


class VoyageAdapter:
    """Adapter for the Voyage embeddings API (MongoDB-hosted endpoint).

    Args:
        model: Voyage model identifier (e.g. "voyage-4-large").
        dimension: Output embedding dimension. A native width >= this value is
            requested, then truncated and re-normalized to ``dimension``.
        api_key: Voyage API key. Falls back to the VOYAGE_API_KEY env var.
        base_url: API base URL. Falls back to the VOYAGE_BASE_URL env var, then
            to the MongoDB-hosted default.
    """

    def __init__(
        self,
        model: str = "voyage-4-large",
        dimension: int = 384,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._dimension = dimension
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Voyage API key required. Set VOYAGE_API_KEY env var "
                "or pass api_key parameter."
            )
        base = base_url or os.environ.get("VOYAGE_BASE_URL", "") or _DEFAULT_BASE_URL
        self._base_url = base.rstrip("/")
        self._output_dimension = _native_output_dimension(dimension)
        interval_env = os.environ.get("VOYAGE_MIN_INTERVAL")
        self._min_interval = (
            float(interval_env) if interval_env is not None else _DEFAULT_MIN_INTERVAL
        )
        self._last_request_ts = 0.0

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts to embeddings via the Voyage API.

        Args:
            texts: Input strings to encode.
            batch_size: Ignored (uses fixed 128-text Voyage batches).
            normalize: Whether to L2-normalize the output vectors.

        Returns:
            Float32 array of shape (N, D) where D == self.dimension.
        """
        return self._encode_with_input_type(
            texts,
            input_type="document",
            normalize=normalize,
        )

    def encode_queries(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode query texts with Voyage's query input type."""
        return self._encode_with_input_type(
            texts,
            input_type="query",
            normalize=normalize,
        )

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode corpus texts with Voyage's document input type."""
        return self._encode_with_input_type(
            texts,
            input_type="document",
            normalize=normalize,
        )

    def _encode_with_input_type(
        self,
        texts: list[str],
        input_type: str,
        normalize: bool,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        cached = load_cache(f"{self.name}:{input_type}", self.dimension, texts)
        if cached is not None:
            return cached

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            result = self._embed_batch(batch, input_type=input_type)
            all_embeddings.extend(item["embedding"] for item in result["data"])

        arr = np.array(all_embeddings, dtype=np.float32)

        # Truncate to the requested dimension (only ever shrinks; a requested
        # width wider than what was returned is rejected, not silently kept).
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
        """POST one batch to the Voyage embeddings endpoint with bounded retry.

        Retries transient network errors with exponential backoff. On an HTTP
        error, the response body (which carries Voyage's error message) is read
        and surfaced in the raised exception.
        """
        body = {
            "input": batch,
            "model": self._model,
            "input_type": input_type,
            "output_dimension": self._output_dimension,
        }
        data = json.dumps(body).encode()
        url = f"{self._base_url}/embeddings"

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            self._throttle()
            req = urllib.request.Request(url, data=data, method="POST")
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
                # 429: pause long enough to clear the per-minute window.
                if exc.code == 429:
                    time.sleep(_RATE_LIMIT_BACKOFF)
                    continue
            except urllib.error.URLError as exc:
                last_exc = exc
                if attempt == _MAX_RETRIES - 1:
                    raise
            time.sleep(0.5 * (2**attempt))
        # Unreachable: the loop either returns or raises.
        raise last_exc  # type: ignore[misc]

    def _throttle(self) -> None:
        """Sleep so consecutive requests respect the minimum interval."""
        if self._min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_ts
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self._dimension

    @property
    def name(self) -> str:
        """Human-readable model name for leaderboard display."""
        return self._model
