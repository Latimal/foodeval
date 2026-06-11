"""Google Gemini embedding adapter (Gemini API direct).

Uses the Gemini batchEmbedContents REST API directly (not the google SDK).
Requires a GEMINI_API_KEY environment variable (or an api_key argument).

gemini-embedding-2 supports native MRL output dimensions up to 3072, so the
requested dimension is passed as outputDimensionality rather than truncated
client-side. Queries and documents use the RETRIEVAL_QUERY and
RETRIEVAL_DOCUMENT task types respectively.

Usage:
    >>> from foodeval.adapters.gemini_adapter import GeminiAdapter
    >>> adapter = GeminiAdapter("gemini-embedding-2", dimension=384)  # doctest: +SKIP
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

# Texts per batchEmbedContents request. Override with GEMINI_BATCH_SIZE for
# rate-limited tiers where smaller batches fit the remaining per-minute quota.
_BATCH_SIZE = 100

# Per-request timeout (seconds) and bounded retry for transient errors.
# Override retry depth with GEMINI_MAX_RETRIES to survive long quota droughts.
_REQUEST_TIMEOUT = 30
_MAX_RETRIES = 8

# Minimum seconds between requests. Defaults to no throttling; override with
# the GEMINI_MIN_INTERVAL env var for rate-limited tiers.
_DEFAULT_MIN_INTERVAL = 0.0

# Backoff (seconds) applied after a 429 before the next retry. The Gemini
# rate-limit window is one minute, but free-tier quota storms can span
# several windows, so the pause exceeds a full window and retries are deep.
_RATE_LIMIT_BACKOFF = 65.0

# Native output dimension of gemini-embedding-2.
_NATIVE_DIM = 3072

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiAdapter:
    """Adapter for the Gemini embeddings API (direct, not the google SDK).

    Args:
        model: Gemini model identifier (e.g. "gemini-embedding-2").
        dimension: Output embedding dimension. Requested natively via
            outputDimensionality (MRL, max 3072). None uses the native 3072.
        api_key: Gemini API key. Falls back to the GEMINI_API_KEY env var.
    """

    def __init__(
        self,
        model: str = "gemini-embedding-2",
        dimension: int | None = 384,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._dimension = dimension
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY env var "
                "or pass api_key parameter."
            )
        interval_env = os.environ.get("GEMINI_MIN_INTERVAL")
        self._min_interval = (
            float(interval_env) if interval_env is not None else _DEFAULT_MIN_INTERVAL
        )
        self._batch_size = int(os.environ.get("GEMINI_BATCH_SIZE", _BATCH_SIZE))
        self._max_retries = int(os.environ.get("GEMINI_MAX_RETRIES", _MAX_RETRIES))
        self._last_request_ts = 0.0

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts to embeddings via the Gemini API.

        Args:
            texts: Input strings to encode.
            batch_size: Ignored (uses fixed 100-text Gemini batches).
            normalize: Whether to L2-normalize the output vectors.

        Returns:
            Float32 array of shape (N, D) where D == self.dimension.
        """
        return self._encode_with_task_type(
            texts,
            task_type="RETRIEVAL_DOCUMENT",
            normalize=normalize,
        )

    def encode_queries(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode query texts with Gemini's retrieval-query task type."""
        return self._encode_with_task_type(
            texts,
            task_type="RETRIEVAL_QUERY",
            normalize=normalize,
        )

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode corpus texts with Gemini's retrieval-document task type."""
        return self._encode_with_task_type(
            texts,
            task_type="RETRIEVAL_DOCUMENT",
            normalize=normalize,
        )

    def _encode_with_task_type(
        self,
        texts: list[str],
        task_type: str,
        normalize: bool,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        cached = load_cache(f"{self.name}:{task_type}", self.dimension, texts)
        if cached is not None:
            return cached

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            result = self._embed_batch(batch, task_type=task_type)
            all_embeddings.extend(item["values"] for item in result["embeddings"])

        arr = np.array(all_embeddings, dtype=np.float32)

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
        save_cache(f"{self.name}:{task_type}", self.dimension, texts, arr)
        return arr

    def _embed_batch(self, batch: list[str], task_type: str) -> dict:
        """POST one batch to the Gemini batchEmbedContents endpoint with bounded retry.

        Retries transient network errors with exponential backoff. On an HTTP
        error, the response body (which carries Gemini's error message) is read
        and surfaced in the raised exception.
        """
        requests_payload: list[dict] = []
        for text in batch:
            item: dict = {
                "model": f"models/{self._model}",
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
            }
            if self._dimension is not None:
                item["outputDimensionality"] = self._dimension
            requests_payload.append(item)
        body = {"requests": requests_payload}
        data = json.dumps(body).encode()
        url = f"{_BASE_URL}/models/{self._model}:batchEmbedContents"

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            self._throttle()
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Accept", "application/json")
            req.add_header("x-goog-api-key", self._api_key)

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
                if attempt == self._max_retries - 1:
                    raise last_exc from exc
                # 429: pause long enough to clear the per-minute window.
                if exc.code == 429:
                    time.sleep(_RATE_LIMIT_BACKOFF)
                    continue
            except urllib.error.URLError as exc:
                last_exc = exc
                if attempt == self._max_retries - 1:
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
        if self._dimension is not None:
            return self._dimension
        return _NATIVE_DIM

    @property
    def name(self) -> str:
        """Human-readable model name for leaderboard display."""
        if self._dimension is not None:
            return f"{self._model}-{self._dimension}d"
        return self._model
