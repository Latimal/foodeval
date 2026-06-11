"""Adapter factory shared by the CLI and matrix runner."""

from __future__ import annotations

from typing import Any

from foodeval.adapters.base import EmbeddingAdapter


def build_adapter(model_str: str, dim: int | None) -> EmbeddingAdapter:
    """Parse a model string and return the appropriate adapter.

    Model string formats:
        - ``lexical-tf`` (alias ``bm25``) -> BM25Adapter
        - ``openai:MODEL_NAME`` -> OpenAIAdapter
        - ``cohere:MODEL_NAME`` -> CohereAdapter
        - ``voyage:MODEL_NAME`` -> VoyageAdapter
        - ``gemini:MODEL_NAME`` -> GeminiAdapter
        - ``vertex:MODEL_NAME`` -> VertexAdapter
        - ``bedrock:MODEL_ID`` -> BedrockAdapter
        - anything else -> SentenceTransformerAdapter
    """
    if model_str in ("lexical-tf", "bm25"):
        try:
            from foodeval.adapters.bm25 import BM25Adapter
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for the BM25 adapter. "
                "Install with: pip install 'foodeval'"
            ) from exc
        return BM25Adapter()

    if model_str.startswith("openai:"):
        model_name = model_str[len("openai:") :]
        try:
            from foodeval.adapters.openai_adapter import OpenAIAdapter
        except ImportError as exc:
            raise ImportError(
                "openai is required for the OpenAI adapter. "
                "Install with: pip install 'foodeval[api]'"
            ) from exc
        kwargs: dict[str, Any] = {"model": model_name}
        if dim is not None:
            kwargs["dimension"] = dim
        return OpenAIAdapter(**kwargs)

    if model_str.startswith("cohere:"):
        model_name = model_str[len("cohere:") :]
        try:
            from foodeval.adapters.cohere_adapter import CohereAdapter
        except ImportError as exc:
            raise ImportError(
                "cohere adapter dependencies are required. "
                "Reinstall the package: pip install foodeval"
            ) from exc
        kwargs = {"model": model_name}
        if dim is not None:
            kwargs["dimension"] = dim
        return CohereAdapter(**kwargs)

    if model_str.startswith("voyage:"):
        model_name = model_str[len("voyage:") :]
        try:
            from foodeval.adapters.voyage_adapter import VoyageAdapter
        except ImportError as exc:
            raise ImportError(
                "voyage adapter dependencies are required. "
                "Reinstall the package: pip install foodeval"
            ) from exc
        kwargs = {"model": model_name}
        if dim is not None:
            kwargs["dimension"] = dim
        return VoyageAdapter(**kwargs)

    if model_str.startswith("gemini:"):
        model_name = model_str[len("gemini:") :]
        try:
            from foodeval.adapters.gemini_adapter import GeminiAdapter
        except ImportError as exc:
            raise ImportError(
                "gemini adapter dependencies are required. "
                "Reinstall the package: pip install foodeval"
            ) from exc
        kwargs = {"model": model_name}
        if dim is not None:
            kwargs["dimension"] = dim
        return GeminiAdapter(**kwargs)

    if model_str.startswith("vertex:"):
        model_name = model_str[len("vertex:") :]
        try:
            from foodeval.adapters.vertex_adapter import VertexAdapter
        except ImportError as exc:
            raise ImportError(
                "vertex adapter dependencies are required. "
                "Reinstall the package: pip install foodeval"
            ) from exc
        kwargs = {"model": model_name}
        if dim is not None:
            kwargs["dimension"] = dim
        return VertexAdapter(**kwargs)

    if model_str.startswith("bedrock:"):
        model_id = model_str[len("bedrock:") :]
        try:
            from foodeval.adapters.bedrock import BedrockAdapter
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for the Bedrock adapter. "
                "Install with: pip install 'foodeval[api]'"
            ) from exc
        kwargs = {"model_id": model_id}
        if dim is not None:
            kwargs["dimension"] = dim
        return BedrockAdapter(**kwargs)

    try:
        from foodeval.adapters.sentence_transformer import SentenceTransformerAdapter
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for local models. "
            "Install with: pip install 'foodeval[local]'"
        ) from exc
    return SentenceTransformerAdapter(model_name_or_path=model_str, truncate_dim=dim)
