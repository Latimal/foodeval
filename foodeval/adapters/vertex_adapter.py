"""Google Vertex AI embedding adapter (publisher models, ADC auth).

Uses the Vertex AI REST endpoints directly (not the google SDK). Two API
methods are supported, resolved automatically from the model name (or
overridden via the api_method argument):

- ``:predict`` -- batched requests (250 texts each). Used by
  gemini-embedding-001 and other publisher embedding models.
- ``:embedContent`` -- one text per request, fanned out across worker
  threads. Used by gemini-embedding-2, which is not served via :predict
  on Vertex (and batchEmbedContents is not routed there either).

Authentication is Application Default Credentials: the access token comes
from ``gcloud auth application-default print-access-token``, so no API key
is needed (run ``gcloud auth application-default login`` first). The
Google Cloud project comes from the GOOGLE_CLOUD_PROJECT env var (or the
project argument).

Both models support native MRL output dimensions up to 3072, so the
requested dimension is passed as outputDimensionality rather than
truncated client-side. Queries and documents use the RETRIEVAL_QUERY and
RETRIEVAL_DOCUMENT task types respectively.

Usage:
    >>> from foodeval.adapters.vertex_adapter import VertexAdapter
    >>> adapter = VertexAdapter("gemini-embedding-001", dimension=384)  # doctest: +SKIP
    >>> embeddings = adapter.encode(["butter chicken", "paneer tikka"])  # doctest: +SKIP
    >>> embeddings.shape  # doctest: +SKIP
    (2, 384)
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from foodeval.adapters.base import load_cache, save_cache

# Texts per :predict request (250 verified working). Override with
# VERTEX_BATCH_SIZE for quota-constrained projects.
_BATCH_SIZE = 250

# Parallel workers for the :embedContent method (one text per request;
# 20 workers verified at ~18 req/s). Override with VERTEX_WORKERS.
_EMBED_CONTENT_WORKERS = 16

# Per-request timeout (seconds) and bounded retry for transient errors.
# Override retry depth with VERTEX_MAX_RETRIES.
_REQUEST_TIMEOUT = 30
_MAX_RETRIES = 6

# Minimum seconds between requests. Defaults to no throttling; override with
# the VERTEX_MIN_INTERVAL env var for rate-limited projects.
_DEFAULT_MIN_INTERVAL = 0.0

# Backoff (seconds) applied after a 429 before the next retry.
_RATE_LIMIT_BACKOFF = 10.0

# ADC access tokens live ~60 minutes; refresh ours after 45 to stay clear
# of expiry mid-batch.
_TOKEN_TTL = 45 * 60

# Native output dimension of gemini-embedding-001.
_NATIVE_DIM = 3072


class VertexAdapter:
    """Adapter for Vertex AI publisher embedding models (direct, not the google SDK).

    Args:
        model: Vertex publisher model identifier (e.g. "gemini-embedding-001").
        dimension: Output embedding dimension. Requested natively via
            outputDimensionality (MRL, max 3072). None uses the native 3072.
        project: Google Cloud project ID. Falls back to the
            GOOGLE_CLOUD_PROJECT env var.
        location: Vertex location. "global" uses aiplatform.googleapis.com;
            other locations use {location}-aiplatform.googleapis.com.
        api_method: "predict" or "embedContent". None auto-resolves from
            the model name: gemini-embedding-2* uses embedContent (it is
            not served via :predict on Vertex), everything else predict.
    """

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        dimension: int | None = 384,
        project: str | None = None,
        location: str = "global",
        api_method: str | None = None,
    ) -> None:
        self._model = model
        self._dimension = dimension
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        if not self._project:
            raise ValueError(
                "Google Cloud project required. Set GOOGLE_CLOUD_PROJECT "
                "env var or pass project parameter."
            )
        self._location = location
        if api_method is not None:
            self._api_method = api_method
        elif model.startswith("gemini-embedding-2"):
            self._api_method = "embedContent"
        else:
            self._api_method = "predict"
        host = (
            "aiplatform.googleapis.com"
            if location == "global"
            else f"{location}-aiplatform.googleapis.com"
        )
        self._url = (
            f"https://{host}/v1/projects/{self._project}/locations/{location}"
            f"/publishers/google/models/{self._model}:{self._api_method}"
        )
        interval_env = os.environ.get("VERTEX_MIN_INTERVAL")
        self._min_interval = (
            float(interval_env) if interval_env is not None else _DEFAULT_MIN_INTERVAL
        )
        self._batch_size = int(os.environ.get("VERTEX_BATCH_SIZE", _BATCH_SIZE))
        self._max_retries = int(os.environ.get("VERTEX_MAX_RETRIES", _MAX_RETRIES))
        self._workers = int(os.environ.get("VERTEX_WORKERS", _EMBED_CONTENT_WORKERS))
        self._last_request_ts = 0.0
        self._token: str | None = None
        self._token_ts = 0.0
        self._token_lock = threading.Lock()

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts to embeddings via the Vertex AI API.

        Args:
            texts: Input strings to encode.
            batch_size: Ignored (uses fixed 250-text Vertex batches).
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
        """Encode query texts with Vertex's retrieval-query task type."""
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
        """Encode corpus texts with Vertex's retrieval-document task type."""
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

        if self._api_method == "embedContent":
            # One text per request; fan out across threads. executor.map
            # yields results in input order regardless of completion order.
            with ThreadPoolExecutor(max_workers=self._workers) as executor:
                all_embeddings = list(
                    executor.map(lambda t: self._embed_single(t, task_type), texts)
                )
        else:
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                result = self._embed_batch(batch, task_type=task_type)
                all_embeddings.extend(
                    pred["embeddings"]["values"] for pred in result["predictions"]
                )

        arr = np.array(all_embeddings, dtype=np.float32)

        # Vertex may return unnormalized vectors at truncated dimensions, so
        # always normalize client-side when requested.
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
        """Embed up to _BATCH_SIZE texts via one :predict request."""
        instances = [{"content": text, "task_type": task_type} for text in batch]
        body: dict = {"instances": instances}
        if self._dimension is not None:
            body["parameters"] = {"outputDimensionality": self._dimension}
        return self._post(json.dumps(body).encode())

    def _embed_single(self, text: str, task_type: str) -> list[float]:
        """Embed one text via one :embedContent request (no batch support)."""
        body: dict = {
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
        }
        if self._dimension is not None:
            body["outputDimensionality"] = self._dimension
        return self._post(json.dumps(body).encode())["embedding"]["values"]

    def _post(self, data: bytes) -> dict:
        """POST a JSON payload to the model endpoint with bounded retry.

        Retries transient network errors with exponential backoff. On an HTTP
        error, the response body (which carries Vertex's error message) is read
        and surfaced in the raised exception. A 401 invalidates the cached
        access token and retries once without consuming a retry attempt.
        """
        last_exc: Exception | None = None
        attempt = 0
        auth_refreshed = False
        while attempt < self._max_retries:
            self._throttle()
            req = urllib.request.Request(self._url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Accept", "application/json")
            req.add_header("Authorization", f"Bearer {self._get_token()}")

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
                # 401: the cached token likely expired early. Refresh once
                # and retry without consuming a retry attempt.
                if exc.code == 401 and not auth_refreshed:
                    auth_refreshed = True
                    self._invalidate_token()
                    continue
                # 4xx (except 429) are not transient; fail fast.
                if exc.code != 429 and 400 <= exc.code < 500:
                    raise last_exc from exc
                if attempt == self._max_retries - 1:
                    raise last_exc from exc
                # 429: pause before the next retry.
                if exc.code == 429:
                    time.sleep(_RATE_LIMIT_BACKOFF)
                    attempt += 1
                    continue
            except urllib.error.URLError as exc:
                last_exc = exc
                if attempt == self._max_retries - 1:
                    raise
            time.sleep(0.5 * (2**attempt))
            attempt += 1
        # Unreachable: the loop either returns or raises.
        raise last_exc  # type: ignore[misc]

    def _get_token(self) -> str:
        """Return a cached ADC access token, refreshing via gcloud when stale.

        Thread-safe: embedContent worker threads share the cache, and the
        lock ensures only one thread runs the gcloud refresh.
        """
        with self._token_lock:
            if (
                self._token is not None
                and time.monotonic() - self._token_ts < _TOKEN_TTL
            ):
                return self._token
            try:
                result = subprocess.run(
                    ["gcloud", "auth", "application-default", "print-access-token"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                raise RuntimeError(
                    "Failed to obtain a Google Cloud access token via gcloud. "
                    "Run `gcloud auth application-default login` and retry."
                ) from exc
            self._token = result.stdout.strip()
            self._token_ts = time.monotonic()
            return self._token

    def _invalidate_token(self) -> None:
        """Drop the cached access token so the next request fetches a fresh one."""
        with self._token_lock:
            self._token = None
            self._token_ts = 0.0

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
        """Human-readable model name for leaderboard display.

        Intentionally omits any "vertex" prefix so results and the embedding
        cache stay transport-agnostic (same model via Gemini API or Vertex
        shares one identity).
        """
        if self._dimension is not None:
            return f"{self._model}-{self._dimension}d"
        return self._model
