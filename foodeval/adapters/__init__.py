"""Model adapters for FoodEval.

Provides a uniform interface for encoding texts with different embedding
backends: local SentenceTransformer models, AWS Bedrock, OpenAI API,
and a BM25 sparse baseline.

Core adapters are imported directly. API-dependent adapters use factory
functions with lazy imports so that users only need to install the
dependencies they actually use.

Usage:
    >>> from foodeval.adapters import get_sentence_transformer_adapter
    >>> adapter = get_sentence_transformer_adapter("BAAI/bge-m3", truncate_dim=384)  # doctest: +SKIP

    >>> from foodeval.adapters import get_bedrock_adapter
    >>> adapter = get_bedrock_adapter("cohere.embed-multilingual-v3")  # doctest: +SKIP

    >>> from foodeval.adapters import get_openai_adapter
    >>> adapter = get_openai_adapter("text-embedding-3-large", dimension=256)  # doctest: +SKIP

    >>> from foodeval.adapters import get_bm25_adapter
    >>> bm25 = get_bm25_adapter()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodeval.adapters.base import EmbeddingAdapter

if TYPE_CHECKING:
    from foodeval.adapters.bedrock import BedrockAdapter
    from foodeval.adapters.bm25 import BM25Adapter
    from foodeval.adapters.cohere_adapter import CohereAdapter
    from foodeval.adapters.gemini_adapter import GeminiAdapter
    from foodeval.adapters.openai_adapter import OpenAIAdapter
    from foodeval.adapters.sentence_transformer import SentenceTransformerAdapter
    from foodeval.adapters.vertex_adapter import VertexAdapter
    from foodeval.adapters.voyage_adapter import VoyageAdapter


def get_bedrock_adapter(*args, **kwargs) -> BedrockAdapter:
    """Create a BedrockAdapter with lazy import.

    Requires boto3. Install with: pip install 'foodeval[api]'

    All arguments are forwarded to BedrockAdapter.__init__.
    """
    from foodeval.adapters.bedrock import BedrockAdapter

    return BedrockAdapter(*args, **kwargs)


def get_openai_adapter(*args, **kwargs) -> OpenAIAdapter:
    """Create an OpenAIAdapter with lazy import.

    Requires the openai package. Install with: pip install 'foodeval[api]'

    All arguments are forwarded to OpenAIAdapter.__init__.
    """
    from foodeval.adapters.openai_adapter import OpenAIAdapter

    return OpenAIAdapter(*args, **kwargs)


def get_sentence_transformer_adapter(*args, **kwargs) -> SentenceTransformerAdapter:
    """Create a SentenceTransformerAdapter with lazy import.

    Requires sentence-transformers and PyTorch.
    Install with: pip install 'foodeval[local]'

    All arguments are forwarded to SentenceTransformerAdapter.__init__.
    """
    from foodeval.adapters.sentence_transformer import SentenceTransformerAdapter

    return SentenceTransformerAdapter(*args, **kwargs)


def get_bm25_adapter(*args, **kwargs) -> BM25Adapter:
    """Create a BM25Adapter with lazy import.

    Uses scikit-learn (a core dependency). No extra packages needed.

    All arguments are forwarded to BM25Adapter.__init__.
    """
    from foodeval.adapters.bm25 import BM25Adapter

    return BM25Adapter(*args, **kwargs)


def get_cohere_adapter(*args, **kwargs) -> CohereAdapter:
    """Create a CohereAdapter with lazy import.

    Uses only stdlib (urllib). Set COHERE_API_KEY env var or pass api_key.

    All arguments are forwarded to CohereAdapter.__init__.
    """
    from foodeval.adapters.cohere_adapter import CohereAdapter

    return CohereAdapter(*args, **kwargs)


def get_gemini_adapter(*args, **kwargs) -> GeminiAdapter:
    """Create a GeminiAdapter with lazy import.

    Uses only stdlib (urllib). Set GEMINI_API_KEY env var or pass api_key.

    All arguments are forwarded to GeminiAdapter.__init__.
    """
    from foodeval.adapters.gemini_adapter import GeminiAdapter

    return GeminiAdapter(*args, **kwargs)


def get_vertex_adapter(*args, **kwargs) -> VertexAdapter:
    """Create a VertexAdapter with lazy import.

    Uses only stdlib (urllib + subprocess). Auth is Application Default
    Credentials via gcloud (run `gcloud auth application-default login`);
    set GOOGLE_CLOUD_PROJECT env var or pass project.

    All arguments are forwarded to VertexAdapter.__init__.
    """
    from foodeval.adapters.vertex_adapter import VertexAdapter

    return VertexAdapter(*args, **kwargs)


def get_voyage_adapter(*args, **kwargs) -> VoyageAdapter:
    """Create a VoyageAdapter with lazy import.

    Uses only stdlib (urllib). Set VOYAGE_API_KEY env var or pass api_key.
    Set VOYAGE_BASE_URL or pass base_url to override the endpoint.

    All arguments are forwarded to VoyageAdapter.__init__.
    """
    from foodeval.adapters.voyage_adapter import VoyageAdapter

    return VoyageAdapter(*args, **kwargs)


__all__ = [
    "EmbeddingAdapter",
    "get_bedrock_adapter",
    "get_bm25_adapter",
    "get_cohere_adapter",
    "get_gemini_adapter",
    "get_openai_adapter",
    "get_sentence_transformer_adapter",
    "get_vertex_adapter",
    "get_voyage_adapter",
]
