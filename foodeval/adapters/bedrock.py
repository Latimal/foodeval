"""AWS Bedrock adapter for Cohere and Amazon Titan embedding models.

Supports multiple model families with different request/response formats:
- Cohere Embed v3/v4: batched (up to 96 texts), input_type parameter
- Amazon Titan Embed v2: single-text, native Matryoshka dimensions

Usage:
    >>> from foodeval.adapters.bedrock import BedrockAdapter
    >>> adapter = BedrockAdapter("cohere.embed-multilingual-v3", dimension=384)  # doctest: +SKIP
    >>> embeddings = adapter.encode(["butter chicken", "paneer tikka"])  # doctest: +SKIP
    >>> embeddings.shape  # doctest: +SKIP
    (2, 384)
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from foodeval.adapters.base import load_cache, save_cache

try:
    import boto3
    from botocore.config import Config as _BotoConfig
except ImportError:
    boto3 = None  # type: ignore[assignment]
    _BotoConfig = None  # type: ignore[assignment,misc]

# Maximum texts per Cohere API call
_COHERE_BATCH_SIZE = 96

# Network timeouts and bounded retry for transient Bedrock errors.
_READ_TIMEOUT = 60
_CONNECT_TIMEOUT = 10
_MAX_RETRIES = 3


class BedrockAdapter:
    """Adapter for AWS Bedrock embedding models.

    Args:
        model_id: Bedrock model identifier. Supported:
            - "cohere.embed-multilingual-v3"
            - "cohere.embed-english-v3"
            - "cohere.embed-v4:0"
            - "amazon.titan-embed-text-v2:0"
        region: AWS region for the Bedrock endpoint.
        dimension: Output embedding dimension. For Cohere models, embeddings
            are truncated and re-normalized after the API call. For Titan,
            the dimension is passed natively in the request.
    """

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        dimension: int | None = None,
    ) -> None:
        if boto3 is None:
            raise ImportError(
                "boto3 is required for BedrockAdapter. "
                "Install it with: pip install 'foodeval[api]'"
            )

        # Cohere embed-v4 requires an inference profile on Bedrock.
        # Auto-resolve bare model ID to the global inference profile.
        if model_id == "cohere.embed-v4:0":
            model_id = "global.cohere.embed-v4:0"

        self._model_id = model_id
        self._region = os.environ.get("AWS_BEDROCK_REGION", region)
        self._dimension = dimension
        client_kwargs: dict = {"region_name": self._region}
        if _BotoConfig is not None:
            client_kwargs["config"] = _BotoConfig(
                read_timeout=_READ_TIMEOUT,
                connect_timeout=_CONNECT_TIMEOUT,
                retries={"max_attempts": _MAX_RETRIES, "mode": "adaptive"},
            )
        self._client = boto3.client("bedrock-runtime", **client_kwargs)
        self._is_cohere = "cohere." in model_id
        self._is_titan = "amazon.titan" in model_id

        if not self._is_cohere and not self._is_titan:
            raise ValueError(
                f"Unsupported model family: {model_id}. "
                "Expected a cohere.embed-* or amazon.titan-embed-* model ID "
                "(optionally prefixed with 'global.' for inference profiles)."
            )

        # Bedrock embeddings can only be truncated down to a native width,
        # never padded up. Reject a requested dimension that exceeds the
        # model's native size rather than silently returning a narrower array.
        if dimension is not None:
            native = self._native_dimension()
            if dimension > native:
                raise ValueError(
                    f"Requested dimension {dimension} exceeds the native "
                    f"dimension {native} of {model_id}. Choose a value "
                    f"<= {native}."
                )

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts to embeddings via Bedrock API.

        Args:
            texts: Input strings to encode.
            batch_size: Ignored for Cohere (uses fixed 96-text batches).
                Used as batch size for Titan requests.
            normalize: Whether to L2-normalize the output vectors.

        Returns:
            Float32 array of shape (N, D).
        """
        return self._encode_with_role(
            texts,
            batch_size=batch_size,
            normalize=normalize,
            input_type="search_document",
        )

    def encode_queries(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode query texts with a query role where the provider supports it."""
        return self._encode_with_role(
            texts,
            batch_size=batch_size,
            normalize=normalize,
            input_type="search_query",
        )

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode document texts with a document role where the provider supports it."""
        return self._encode_with_role(
            texts,
            batch_size=batch_size,
            normalize=normalize,
            input_type="search_document",
        )

    def _encode_with_role(
        self,
        texts: list[str],
        batch_size: int,
        normalize: bool,
        input_type: str,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        cache_name = f"{self.name}:{input_type}" if self._is_cohere else self.name
        cached = load_cache(cache_name, self.dimension, texts)
        if cached is not None:
            return cached

        if self._is_cohere:
            embeddings = self._encode_cohere(texts, input_type=input_type)
        else:
            embeddings = self._encode_titan(texts, batch_size)

        # Neither Cohere v3 nor Titan v2 (on Bedrock) support native
        # dimension reduction. Truncate and re-normalize manually.
        if self._dimension is not None:
            embeddings = embeddings[:, : self._dimension]

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            embeddings = embeddings / norms

        embeddings = embeddings.astype(np.float32)
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"{self._model_id} returned {embeddings.shape[1]}d embeddings "
                f"but adapter reports dimension {self.dimension}. The model's "
                "native width is smaller than the requested dimension."
            )
        save_cache(cache_name, self.dimension, texts, embeddings)
        return embeddings

    def _invoke(self, body: dict) -> dict:
        """Call invoke_model with a bounded retry and exponential backoff.

        botocore handles throttling retries at the SDK layer; this adds a
        backstop for transient connection/read-timeout errors. The final
        attempt's exception propagates unchanged.
        """
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.invoke_model(
                    modelId=self._model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
                return json.loads(response["body"].read())
            except Exception as exc:  # noqa: BLE001 - re-raised after retries
                last_exc = exc
                if attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(0.5 * (2**attempt))
        # Unreachable: the loop either returns or raises.
        raise last_exc  # type: ignore[misc]

    def _encode_cohere(self, texts: list[str], input_type: str) -> np.ndarray:
        """Encode texts using a Cohere Embed model.

        Batches texts into groups of up to 96, with rate limiting between
        batches.
        """
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), _COHERE_BATCH_SIZE):
            batch = texts[i : i + _COHERE_BATCH_SIZE]
            body = {
                "texts": batch,
                "input_type": input_type,
                "truncate": "END",
            }
            result = self._invoke(body)
            # Cohere v4 nests embeddings under {"float": [...]},
            # while v3 returns a flat list directly.
            embeddings = result["embeddings"]
            if isinstance(embeddings, dict):
                embeddings = embeddings["float"]
            all_embeddings.extend(embeddings)

            # Rate limiting between batches
            if i + _COHERE_BATCH_SIZE < len(texts):
                time.sleep(0.1)

        return np.array(all_embeddings, dtype=np.float32)

    def _encode_titan(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Encode texts using Amazon Titan Embed.

        Titan accepts only one text per request, so we loop with rate
        limiting between batches.
        """
        all_embeddings: list[list[float]] = []

        # Titan v2 only accepts dimensions in {256, 512, 1024}.
        # For non-standard values (e.g. 384), request the next-larger
        # valid dimension and let encode() truncate afterwards.
        titan_dims = None
        if self._dimension is not None:
            for valid in (256, 512, 1024):
                if self._dimension <= valid:
                    titan_dims = valid
                    break

        for i, text in enumerate(texts):
            body: dict = {
                "inputText": text,
                "normalize": True,
            }
            if titan_dims is not None:
                body["dimensions"] = titan_dims

            result = self._invoke(body)
            all_embeddings.append(result["embedding"])

            # Rate limiting: sleep after every `batch_size` texts
            if (i + 1) % batch_size == 0 and (i + 1) < len(texts):
                time.sleep(0.1)

        return np.array(all_embeddings, dtype=np.float32)

    def _native_dimension(self) -> int:
        """Native (un-truncated) embedding width for the configured model."""
        if "embed-v4" in self._model_id:
            return 1536
        if self._is_cohere:
            return 1024
        # Titan default
        return 1024

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        if self._dimension is not None:
            return self._dimension
        return self._native_dimension()

    @property
    def name(self) -> str:
        """Human-readable model name for leaderboard display."""
        # Strip version suffixes for readability
        base = self._model_id.replace(":0", "").split(".")[-1]
        if self._dimension is not None:
            return f"{base}-{self._dimension}d"
        return base
