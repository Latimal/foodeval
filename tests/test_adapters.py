"""Tests for embedding adapters: protocol compliance, mock adapter, BM25, and API adapters (FoodEval).

API-dependent adapters (SentenceTransformer, Bedrock, OpenAI, Cohere) are
tested two ways: construction/error handling, and encode() behavior with the
network client mocked at the boundary (no real API calls, no GPU, no model
downloads). The encode() tests verify response parsing, index reordering,
batching limits, dimension handling, and L2 normalization.
"""

from __future__ import annotations

import json
import time
import urllib.error
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from foodeval.adapters.base import (
    EmbeddingAdapter,
    _sanitize_adapter_name,
    cache_key,
    load_cache,
    save_cache,
)


# =========================================================================
# _sanitize_adapter_name
# =========================================================================


class TestSanitizeAdapterName:
    """_sanitize_adapter_name: prevent path traversal in cache file paths."""

    def test_path_traversal_stripped(self):
        """Directory traversal sequences should not survive sanitization."""
        result = _sanitize_adapter_name("../../etc/passwd")
        assert "/" not in result
        assert ".." not in result

    def test_forward_slashes_replaced(self):
        """Forward slashes in org/model names should become underscores."""
        assert _sanitize_adapter_name("org/model-name") == "org_model-name"

    def test_backslashes_replaced(self):
        """Backslashes (Windows paths) should become underscores."""
        assert _sanitize_adapter_name("path\\model") == "path_model"

    def test_clean_name_passes_through(self):
        """A simple model name with no separators should be unchanged."""
        assert _sanitize_adapter_name("bge-m3") == "bge-m3"


# =========================================================================
# DummyAdapter protocol compliance
# =========================================================================


class TestDummyAdapterProtocol:
    """Verify the test DummyAdapter satisfies EmbeddingAdapter protocol."""

    def test_satisfies_protocol(self, dummy_adapter):
        """DummyAdapter must be recognized as EmbeddingAdapter at runtime."""
        assert isinstance(dummy_adapter, EmbeddingAdapter)

    def test_encode_returns_correct_shape(self, dummy_adapter):
        texts = ["butter chicken", "paneer tikka", "iced tea"]
        result = dummy_adapter.encode(texts)
        assert result.shape == (3, 64)

    def test_encode_empty_list(self, dummy_adapter):
        result = dummy_adapter.encode([])
        assert result.shape == (0, 64)

    def test_encode_normalized_by_default(self, dummy_adapter):
        result = dummy_adapter.encode(["chicken biryani"])
        norm = np.linalg.norm(result[0])
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_encode_unnormalized(self, dummy_adapter):
        result = dummy_adapter.encode(["chicken biryani"], normalize=False)
        norm = np.linalg.norm(result[0])
        # Random vectors are very unlikely to be exactly unit-length
        assert not np.isclose(norm, 1.0, atol=1e-3)

    def test_same_text_same_embedding(self, dummy_adapter):
        """Identical text should produce identical embeddings."""
        text = "mango lassi"
        r1 = dummy_adapter.encode([text])
        r2 = dummy_adapter.encode([text])
        np.testing.assert_array_equal(r1, r2)

    def test_cross_instance_reproducible(self):
        """A fresh adapter instance must reproduce the exact same embeddings.

        The per-text seed is derived from a stable content hash (crc32), not
        the process-salted built-in hash(), so re-instantiating the adapter
        (or running in a separate process) yields byte-identical vectors. This
        is the contract the 'reproducible' docstring promises and the guard
        against a regression back to hash()-based seeding.
        """
        from tests.conftest import DummyAdapter

        texts = ["butter chicken", "paneer tikka masala", "cold brew coffee"]
        first = DummyAdapter(dim=64).encode(texts)
        second = DummyAdapter(dim=64).encode(texts)
        assert np.array_equal(first, second)

    def test_different_text_different_embedding(self, dummy_adapter):
        """Different texts should produce different embeddings."""
        r1 = dummy_adapter.encode(["green curry"])
        r2 = dummy_adapter.encode(["iced coffee"])
        assert not np.array_equal(r1, r2)

    def test_dimension_property(self, dummy_adapter):
        assert dummy_adapter.dimension == 64

    def test_name_property(self, dummy_adapter):
        assert dummy_adapter.name == "dummy-64d"

    def test_order_independent_embeddings(self, dummy_adapter):
        """Each text's embedding should be independent of the batch it's in."""
        texts_a = ["butter chicken", "iced tea"]
        texts_b = ["iced tea", "butter chicken"]
        r_a = dummy_adapter.encode(texts_a)
        r_b = dummy_adapter.encode(texts_b)
        np.testing.assert_allclose(r_a[0], r_b[1], atol=1e-6)
        np.testing.assert_allclose(r_a[1], r_b[0], atol=1e-6)

    def test_encode_returns_float32(self, dummy_adapter):
        result = dummy_adapter.encode(["paneer butter masala"])
        assert result.dtype == np.float32


class TestConstantAdapterProtocol:
    """Verify the ConstantAdapter satisfies the protocol and behaves as expected."""

    def test_satisfies_protocol(self, constant_adapter):
        assert isinstance(constant_adapter, EmbeddingAdapter)

    def test_all_embeddings_identical(self, constant_adapter):
        """Every text produces the same embedding vector."""
        texts = ["butter chicken", "iced tea", "pad thai"]
        result = constant_adapter.encode(texts)
        np.testing.assert_array_equal(result[0], result[1])
        np.testing.assert_array_equal(result[1], result[2])

    def test_embeddings_are_normalized(self, constant_adapter):
        result = constant_adapter.encode(["green curry"])
        norm = np.linalg.norm(result[0])
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_cosine_similarity_is_one_for_all_pairs(self, constant_adapter):
        """All pairs should have cosine similarity = 1.0."""
        texts = ["pasta", "ramen"]
        embs = constant_adapter.encode(texts)
        cos_sim = np.dot(embs[0], embs[1])
        assert cos_sim == pytest.approx(1.0, abs=1e-5)


# =========================================================================
# BM25 adapter
# =========================================================================


class TestBM25Adapter:
    """BM25 lexical baseline (hashed TF-IDF, EmbeddingAdapter protocol)."""

    @pytest.fixture
    def bm25(self):
        from foodeval.adapters.bm25 import BM25Adapter

        return BM25Adapter(dim=256)

    @pytest.fixture
    def bm25_512(self):
        from foodeval.adapters.bm25 import BM25Adapter

        return BM25Adapter(dim=512)

    def test_satisfies_protocol(self, bm25):
        assert isinstance(bm25, EmbeddingAdapter)

    def test_encode_returns_correct_shape(self, bm25_512):
        result = bm25_512.encode(["butter chicken", "iced tea"])
        assert result.shape == (2, 512)

    def test_encode_default_dim_is_4096(self):
        from foodeval.adapters.bm25 import BM25Adapter

        adapter = BM25Adapter()
        assert adapter.dimension == 4096

    def test_name_property(self, bm25):
        assert bm25.name == "Lexical (TF)"

    def test_encode_returns_float32(self, bm25):
        result = bm25.encode(["green curry"])
        assert result.dtype == np.float32

    def test_overlapping_terms_higher_cosine(self, bm25_512):
        """Texts sharing terms should have higher cosine similarity than unrelated texts."""
        embs = bm25_512.encode(["chicken curry", "butter chicken", "iced tea"])
        cos_related = np.dot(embs[0], embs[1])
        cos_unrelated = np.dot(embs[0], embs[2])
        assert cos_related > cos_unrelated

    def test_encode_normalized_by_default(self, bm25):
        result = bm25.encode(["paneer tikka masala"])
        norm = np.linalg.norm(result[0])
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_different_texts_different_vectors(self, bm25):
        result = bm25.encode(["spaghetti bolognese", "green tea"])
        assert not np.array_equal(result[0], result[1])

    def test_identical_texts_identical_vectors(self, bm25):
        r1 = bm25.encode(["mango lassi"])
        r2 = bm25.encode(["mango lassi"])
        np.testing.assert_array_equal(r1, r2)

    def test_exact_ndcg_on_fixed_corpus(self):
        """Pin the exact NDCG@10 for TF-IDF on a tiny, hand-checked corpus.

        Query "chicken curry" against ["chicken curry", "chicken tikka",
        "mango lassi"]. Cosines: exact match = 1.0, one shared token = 0.5,
        disjoint = 0.0, so the ranking is [curry, tikka, lassi]. With relevance
        grades {curry: 3, tikka: 1, lassi: 0} this IS the ideal ranking, so
        NDCG@10 = 1.0 exactly. A regression that scrambles lexical ranking or
        mishandles graded relevance would drop this below 1.0.
        """
        from foodeval.adapters.bm25 import BM25Adapter
        from foodeval.metrics.ndcg import ndcg_at_k

        adapter = BM25Adapter(dim=4096)
        corpus = ["chicken curry", "chicken tikka", "mango lassi"]
        c_emb = adapter.encode(corpus, normalize=True)
        q_emb = adapter.encode(["chicken curry"], normalize=True)
        sims = (q_emb @ c_emb.T).flatten()
        ranked = np.argsort(-sims, kind="stable")

        relevance = {"chicken curry": 3, "chicken tikka": 1, "mango lassi": 0}
        rel_at_rank = [relevance[corpus[i]] for i in ranked]
        assert rel_at_rank == [3, 1, 0]
        assert ndcg_at_k(rel_at_rank, k=10) == 1.0

    def test_exact_best_f1_on_partial_overlap_pairs(self):
        """Pin the exact best_f1 for TF-IDF on pairs it cannot fully separate.

        A positive pair (paneer tikka / paneer masala, cos 0.5) and a negative
        pair (chicken curry / chicken tikka, cos 0.5) tie, so no threshold
        separates them. The optimal threshold (0.5) accepts both, giving
        tp=2, fp=1, fn=0, tn=1 -> precision 2/3, recall 1.0, F1 = 0.8 exactly.
        This pins the threshold-sweep math, not just trivial separation.
        """
        from foodeval.adapters.bm25 import BM25Adapter
        from foodeval.metrics.f1 import best_f1

        adapter = BM25Adapter(dim=4096)
        texts_a = ["chicken curry rice", "paneer tikka", "chicken curry", "mango lassi"]
        texts_b = ["chicken curry rice", "paneer masala", "chicken tikka", "green tea"]
        labels = [1, 1, 0, 0]
        emb_a = adapter.encode(texts_a, normalize=True)
        emb_b = adapter.encode(texts_b, normalize=True)
        sims = np.sum(emb_a * emb_b, axis=1).tolist()

        result = best_f1(labels, sims)
        assert result["f1"] == pytest.approx(0.8)
        assert result["tp"] == 2
        assert result["fp"] == 1
        assert result["fn"] == 0
        assert result["tn"] == 1


# =========================================================================
# SentenceTransformer adapter: construction-only tests
# =========================================================================


class TestSentenceTransformerAdapterConstruction:
    """SentenceTransformer adapter construction and error handling."""

    def test_import_error_without_library(self):
        """Should raise ImportError with helpful message when library is missing."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            from foodeval.adapters import sentence_transformer

            original_st = sentence_transformer.SentenceTransformer
            sentence_transformer.SentenceTransformer = None
            try:
                with pytest.raises(ImportError, match="sentence-transformers"):
                    sentence_transformer.SentenceTransformerAdapter("fake-model")
            finally:
                sentence_transformer.SentenceTransformer = original_st

    def test_name_includes_truncate_dim(self):
        """When truncate_dim is set, the name should include the dimension."""
        from foodeval.adapters import sentence_transformer

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        with patch.object(
            sentence_transformer,
            "SentenceTransformer",
            return_value=mock_model,
        ):
            adapter = sentence_transformer.SentenceTransformerAdapter(
                "BAAI/bge-m3", truncate_dim=384
            )
            assert adapter.name == "bge-m3-384d"

    def test_name_without_truncate_dim(self):
        """Without truncate_dim, the name is just the model basename."""
        from foodeval.adapters import sentence_transformer

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        with patch.object(
            sentence_transformer,
            "SentenceTransformer",
            return_value=mock_model,
        ):
            adapter = sentence_transformer.SentenceTransformerAdapter("BAAI/bge-m3")
            assert adapter.name == "bge-m3"


# =========================================================================
# Bedrock adapter: construction-only tests
# =========================================================================


class TestBedrockAdapterConstruction:
    """Bedrock adapter construction and error handling."""

    def test_import_error_without_boto3(self):
        """Should raise ImportError when boto3 is not available."""
        from foodeval.adapters import bedrock

        original_boto3 = bedrock.boto3
        bedrock.boto3 = None
        try:
            with pytest.raises(ImportError, match="boto3"):
                bedrock.BedrockAdapter("cohere.embed-multilingual-v3")
        finally:
            bedrock.boto3 = original_boto3

    def test_unsupported_model_family(self):
        """Should reject model IDs that are not cohere or titan."""
        from foodeval.adapters.bedrock import BedrockAdapter

        with patch("foodeval.adapters.bedrock.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            with pytest.raises(ValueError, match="Unsupported model family"):
                BedrockAdapter("anthropic.claude-v3")

    def test_cohere_default_dimension(self):
        """Cohere models default to 1024 dimensions."""
        from foodeval.adapters.bedrock import BedrockAdapter

        with patch("foodeval.adapters.bedrock.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            adapter = BedrockAdapter("cohere.embed-multilingual-v3")
            assert adapter.dimension == 1024

    def test_titan_default_dimension(self):
        """Titan models default to 1024 dimensions."""
        from foodeval.adapters.bedrock import BedrockAdapter

        with patch("foodeval.adapters.bedrock.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            adapter = BedrockAdapter("amazon.titan-embed-text-v2:0")
            assert adapter.dimension == 1024

    def test_custom_dimension(self):
        from foodeval.adapters.bedrock import BedrockAdapter

        with patch("foodeval.adapters.bedrock.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            adapter = BedrockAdapter("cohere.embed-multilingual-v3", dimension=384)
            assert adapter.dimension == 384

    def test_name_with_dimension(self):
        from foodeval.adapters.bedrock import BedrockAdapter

        with patch("foodeval.adapters.bedrock.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            adapter = BedrockAdapter("cohere.embed-multilingual-v3", dimension=384)
            assert adapter.name == "embed-multilingual-v3-384d"

    def test_name_without_dimension(self):
        from foodeval.adapters.bedrock import BedrockAdapter

        with patch("foodeval.adapters.bedrock.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            adapter = BedrockAdapter("cohere.embed-multilingual-v3")
            assert adapter.name == "embed-multilingual-v3"

    def test_titan_name_strips_version_suffix(self):
        from foodeval.adapters.bedrock import BedrockAdapter

        with patch("foodeval.adapters.bedrock.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            adapter = BedrockAdapter("amazon.titan-embed-text-v2:0", dimension=256)
            assert adapter.name == "titan-embed-text-v2-256d"

    def test_requested_dimension_wider_than_native_raises(self):
        """Bedrock can only truncate down. A dimension above native is rejected
        at construction, before any API call."""
        from foodeval.adapters.bedrock import BedrockAdapter

        with patch("foodeval.adapters.bedrock.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            # Cohere v3 native width is 1024.
            with pytest.raises(ValueError, match="exceeds the native"):
                BedrockAdapter("cohere.embed-multilingual-v3", dimension=2048)


# =========================================================================
# Bedrock adapter: encode() behavior with mocked invoke_model
# =========================================================================


def _make_bedrock_adapter(invoke_fn, model_id, dimension=4):
    """Build a BedrockAdapter whose client.invoke_model runs invoke_fn.

    invoke_fn receives the decoded request body dict and returns the decoded
    response dict; this helper wraps the JSON encode/decode and the streaming
    body that botocore returns. Caching is patched out.
    """
    patches = [
        patch("foodeval.adapters.bedrock.boto3"),
        patch("foodeval.adapters.bedrock.load_cache", return_value=None),
        patch("foodeval.adapters.bedrock.save_cache"),
    ]
    mocks = [p.start() for p in patches]
    try:
        mock_boto3 = mocks[0]
        client = MagicMock()
        mock_boto3.client.return_value = client

        def _side_effect(modelId, contentType, accept, body):
            request = json.loads(body)
            response_dict = invoke_fn(modelId, request)
            return {"body": BytesIO(json.dumps(response_dict).encode())}

        client.invoke_model.side_effect = _side_effect
        from foodeval.adapters.bedrock import BedrockAdapter

        adapter = BedrockAdapter(model_id, dimension=dimension)
    except Exception:
        for p in patches:
            p.stop()
        raise
    return adapter, client, patches


class TestBedrockAdapterEncode:
    """BedrockAdapter.encode: Cohere v3/v4 and Titan paths, batching, output shape."""

    def test_cohere_v3_flat_list_response(self):
        """Cohere v3 returns embeddings as a flat list under 'embeddings'."""

        def invoke_fn(model_id, request):
            n = len(request["texts"])
            return {"embeddings": [[float(i + 1)] * 8 for i in range(n)]}

        adapter, client, patches = _make_bedrock_adapter(
            invoke_fn, "cohere.embed-multilingual-v3", dimension=4
        )
        try:
            out = adapter.encode(["butter chicken", "paneer tikka"], normalize=True)
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (2, 4)
        assert out.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-6)
        assert client.invoke_model.call_args.kwargs["modelId"] == (
            "cohere.embed-multilingual-v3"
        )

    def test_cohere_v4_nested_float_response(self):
        """Cohere v4 nests embeddings under {'float': [...]}.

        The adapter must unwrap the 'float' key. The bare v4 model ID is also
        auto-resolved to the global inference profile.
        """

        def invoke_fn(model_id, request):
            n = len(request["texts"])
            return {"embeddings": {"float": [[float(i + 1)] * 8 for i in range(n)]}}

        adapter, client, patches = _make_bedrock_adapter(
            invoke_fn, "cohere.embed-v4:0", dimension=4
        )
        try:
            out = adapter.encode(["x", "y", "z"], normalize=True)
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (3, 4)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.ones(3), atol=1e-6)
        assert client.invoke_model.call_args.kwargs["modelId"] == (
            "global.cohere.embed-v4:0"
        )

    def test_cohere_batches_at_96(self):
        """Cohere calls are capped at 96 texts per request."""
        batch_sizes: list[int] = []

        def invoke_fn(model_id, request):
            batch_sizes.append(len(request["texts"]))
            n = len(request["texts"])
            return {"embeddings": [[1.0] * 8 for _ in range(n)]}

        adapter, client, patches = _make_bedrock_adapter(
            invoke_fn, "cohere.embed-multilingual-v3", dimension=4
        )
        try:
            out = adapter.encode([f"t{i}" for i in range(100)])
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (100, 4)
        assert client.invoke_model.call_count == 2
        assert batch_sizes == [96, 4]

    def test_titan_single_text_per_call(self):
        """Titan accepts one text per request, so N texts => N invocations."""
        requests: list[dict] = []

        def invoke_fn(model_id, request):
            requests.append(request)
            return {"embedding": [1.0] * 256}

        adapter, client, patches = _make_bedrock_adapter(
            invoke_fn, "amazon.titan-embed-text-v2:0", dimension=256
        )
        try:
            out = adapter.encode(["a", "b", "c"], normalize=True)
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (3, 256)
        assert client.invoke_model.call_count == 3
        # Each request carries the single input text and the resolved dimension.
        assert [r["inputText"] for r in requests] == ["a", "b", "c"]
        assert requests[0]["dimensions"] == 256

    def test_titan_truncates_to_requested_dimension(self):
        """A non-standard dim (384) requests the next valid Titan width (512)
        and truncates the result down to 384."""
        requests: list[dict] = []

        def invoke_fn(model_id, request):
            requests.append(request)
            return {"embedding": [0.5] * 512}

        adapter, _client, patches = _make_bedrock_adapter(
            invoke_fn, "amazon.titan-embed-text-v2:0", dimension=384
        )
        try:
            out = adapter.encode(["x"], normalize=True)
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (1, 384)
        assert requests[0]["dimensions"] == 512

    def test_empty_input_returns_empty_array(self):
        """encode([]) returns shape (0, dim) without invoking the model."""

        def invoke_fn(model_id, request):  # pragma: no cover - must not run
            raise AssertionError("invoke_model should not be called for empty input")

        adapter, client, patches = _make_bedrock_adapter(
            invoke_fn, "cohere.embed-multilingual-v3", dimension=4
        )
        try:
            out = adapter.encode([])
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (0, 4)
        client.invoke_model.assert_not_called()


# =========================================================================
# OpenAI adapter: construction-only tests
# =========================================================================


class TestOpenAIAdapterConstruction:
    """OpenAI adapter construction and error handling."""

    def test_import_error_without_library(self):
        from foodeval.adapters import openai_adapter

        original_openai = openai_adapter.openai
        openai_adapter.openai = None
        try:
            with pytest.raises(ImportError, match="openai"):
                openai_adapter.OpenAIAdapter(api_key="test-key")
        finally:
            openai_adapter.openai = original_openai

    def test_missing_api_key_raises(self, monkeypatch):
        """Should raise ValueError when no API key is available."""
        from foodeval.adapters.openai_adapter import OpenAIAdapter

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch("foodeval.adapters.openai_adapter.openai") as mock_openai:
            mock_openai.OpenAI = MagicMock()
            with pytest.raises(ValueError, match="No API key"):
                OpenAIAdapter(api_key=None)

    def test_name_format(self):
        from foodeval.adapters.openai_adapter import OpenAIAdapter

        with patch("foodeval.adapters.openai_adapter.openai") as mock_openai:
            mock_openai.OpenAI = MagicMock()
            adapter = OpenAIAdapter(
                model="text-embedding-3-large",
                dimension=256,
                api_key="test-key-12345",
            )
            assert adapter.name == "text-embedding-3-large-256d"
            assert adapter.dimension == 256

    def test_accepts_explicit_api_key(self):
        """Should not raise when api_key is provided directly."""
        from foodeval.adapters.openai_adapter import OpenAIAdapter

        with patch("foodeval.adapters.openai_adapter.openai") as mock_openai:
            mock_openai.OpenAI = MagicMock()
            # Should not raise
            adapter = OpenAIAdapter(api_key="sk-test-key-for-unit-test")
            assert adapter.dimension == 384  # Default dimension


# =========================================================================
# OpenAI adapter: encode() behavior with mocked client
# =========================================================================


def _make_openai_adapter(create_fn, dimension=4, model="text-embedding-3-large"):
    """Build an OpenAIAdapter whose client.embeddings.create runs create_fn.

    Caching is patched out so encode() always exercises the API path.
    """
    patches = [
        patch("foodeval.adapters.openai_adapter.openai"),
        patch("foodeval.adapters.openai_adapter.load_cache", return_value=None),
        patch("foodeval.adapters.openai_adapter.save_cache"),
    ]
    mocks = [p.start() for p in patches]
    try:
        mock_openai = mocks[0]
        client = MagicMock()
        mock_openai.OpenAI.return_value = client
        client.embeddings.create.side_effect = create_fn
        from foodeval.adapters.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(model=model, dimension=dimension, api_key="sk-test")
    except Exception:
        for p in patches:
            p.stop()
        raise
    return adapter, client, patches


def _openai_response_in_position_order(*, model, input, dimensions):
    """Return a response whose row i has its first component = i.

    The .index values are correct but the data list is REVERSED, so a caller
    that naively reads response.data in order would scramble the rows. A
    correct caller reorders by .index and recovers the input order. Signature
    mirrors the keyword call ``create(model=, input=, dimensions=)``.
    """
    items = [
        SimpleNamespace(index=pos, embedding=[float(pos)] + [0.1] * (dimensions - 1))
        for pos in range(len(input))
    ]
    return SimpleNamespace(data=list(reversed(items)))


class TestOpenAIAdapterEncode:
    """OpenAIAdapter.encode: response parsing, ordering, batching, normalization."""

    def test_rows_land_in_input_order_despite_shuffled_index(self):
        """Output row i must correspond to input i, even when the API returns
        data out of order. The adapter reorders by item.index."""
        adapter, _client, patches = _make_openai_adapter(
            _openai_response_in_position_order, dimension=4
        )
        try:
            out = adapter.encode(["a", "b", "c"], normalize=False)
        finally:
            for p in patches:
                p.stop()
        # First component encodes the original input position.
        assert out[:, 0].tolist() == [0.0, 1.0, 2.0]

    def test_output_is_l2_normalized_float32(self):
        """With normalize=True, every row is unit-length float32 of shape (N, dim)."""
        adapter, _client, patches = _make_openai_adapter(
            _openai_response_in_position_order, dimension=4
        )
        try:
            out = adapter.encode(["x", "y", "z"], normalize=True)
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (3, 4)
        assert out.dtype == np.float32
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, np.ones(3), atol=1e-6)

    def test_batches_over_100_split_into_multiple_calls(self):
        """The per-call cap is 100 texts, even if batch_size is larger."""
        call_sizes: list[int] = []

        def create_fn(model, input, dimensions):
            call_sizes.append(len(input))
            return _openai_response_in_position_order(
                model=model, input=input, dimensions=dimensions
            )

        adapter, client, patches = _make_openai_adapter(create_fn, dimension=4)
        try:
            # normalize=False so the helper's position encoding (first component
            # = within-batch index) survives for the reassembly assertion below.
            out = adapter.encode(
                [f"item{i}" for i in range(150)], batch_size=1000, normalize=False
            )
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (150, 4)
        assert client.embeddings.create.call_count == 2
        assert call_sizes == [100, 50]
        # Cross-batch reassembly: the helper sets each row's first component to
        # its within-batch position, so the second batch (global rows 100-149)
        # must restart at 0. This catches an off-by-batch placement bug.
        assert out[0, 0] == 0.0
        assert out[99, 0] == 99.0
        assert out[100, 0] == 0.0
        assert out[149, 0] == 49.0

    def test_calls_create_with_model_and_dimensions(self):
        """The API request must carry the configured model and dimensions."""
        adapter, client, patches = _make_openai_adapter(
            _openai_response_in_position_order, dimension=256
        )
        try:
            adapter.encode(["butter chicken", "paneer tikka"])
        finally:
            for p in patches:
                p.stop()
        _, kwargs = client.embeddings.create.call_args
        assert kwargs["model"] == "text-embedding-3-large"
        assert kwargs["dimensions"] == 256
        assert kwargs["input"] == ["butter chicken", "paneer tikka"]

    def test_empty_input_returns_empty_array(self):
        """encode([]) must return shape (0, dim) without calling the API."""
        adapter, client, patches = _make_openai_adapter(
            _openai_response_in_position_order, dimension=384
        )
        try:
            out = adapter.encode([])
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (0, 384)
        client.embeddings.create.assert_not_called()

    def test_width_mismatch_raises_value_error(self):
        """If the API returns a different width than requested, raise ValueError.

        Guards against silently accepting a model that ignored the
        'dimensions' parameter.
        """

        def create_fn(model, input, dimensions):
            # Return 8-wide rows while the adapter expects 4.
            items = [
                SimpleNamespace(index=pos, embedding=[1.0] * 8)
                for pos in range(len(input))
            ]
            return SimpleNamespace(data=items)

        adapter, _client, patches = _make_openai_adapter(create_fn, dimension=4)
        try:
            with pytest.raises(ValueError, match="returned 8d embeddings"):
                adapter.encode(["a", "b"])
        finally:
            for p in patches:
                p.stop()


# =========================================================================
# Cohere adapter: construction-only tests
# =========================================================================


class TestCohereAdapterConstruction:
    """Cohere adapter construction and error handling."""

    def test_missing_api_key_raises(self, monkeypatch):
        """Should raise ValueError when no API key is available."""
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        from foodeval.adapters.cohere_adapter import CohereAdapter

        with pytest.raises(ValueError, match="Cohere API key"):
            CohereAdapter("embed-multilingual-v3.0", api_key=None)

    def test_name_format_includes_dimension(self):
        """When dimension is set, the name should include it."""
        from foodeval.adapters.cohere_adapter import CohereAdapter

        adapter = CohereAdapter(
            "embed-multilingual-v3.0",
            dimension=384,
            api_key="test-key-for-unit-test",
        )
        assert adapter.name == "embed-multilingual-v3-384d"

    def test_native_dimension_from_known_models(self):
        """Known models should return their native dimension when no override."""
        from foodeval.adapters.cohere_adapter import CohereAdapter

        adapter = CohereAdapter(
            "embed-multilingual-v3.0",
            api_key="test-key-for-unit-test",
        )
        assert adapter.dimension == 1024


# =========================================================================
# Cohere adapter: encode() behavior with mocked urlopen
# =========================================================================


def _make_cohere_adapter(urlopen_fn, dimension=4, model="embed-multilingual-v3.0"):
    """Build a CohereAdapter whose urllib urlopen runs urlopen_fn.

    urlopen_fn(request) returns a file-like object whose .read() yields the
    JSON response bytes. Caching is patched out.
    """
    patches = [
        patch(
            "foodeval.adapters.cohere_adapter.urllib.request.urlopen",
            side_effect=urlopen_fn,
        ),
        patch("foodeval.adapters.cohere_adapter.load_cache", return_value=None),
        patch("foodeval.adapters.cohere_adapter.save_cache"),
    ]
    for p in patches:
        p.start()
    try:
        from foodeval.adapters.cohere_adapter import CohereAdapter

        adapter = CohereAdapter(model, dimension=dimension, api_key="test-key-xyz")
    except Exception:
        for p in patches:
            p.stop()
        raise
    return adapter, patches


class TestCohereAdapterEncode:
    """CohereAdapter.encode: request shape, headers, parsing, batching, normalization."""

    def test_posts_to_v2_embed_with_auth_and_body(self):
        """The request hits the v2 embed endpoint with a bearer token and the
        expected JSON body (model, texts, input_type, embedding_types)."""
        captured: dict = {}

        def urlopen_fn(req, timeout=None):
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            captured["headers"] = {k.lower(): v for k, v in req.header_items()}
            captured["body"] = json.loads(req.data.decode())
            n = len(captured["body"]["texts"])
            payload = {"embeddings": {"float": [[float(i + 1)] * 8 for i in range(n)]}}
            return BytesIO(json.dumps(payload).encode())

        adapter, patches = _make_cohere_adapter(urlopen_fn, dimension=4)
        try:
            adapter.encode(["butter chicken", "paneer tikka"])
        finally:
            for p in patches:
                p.stop()
        assert captured["url"] == "https://api.cohere.com/v2/embed"
        assert captured["method"] == "POST"
        assert captured["headers"]["authorization"] == "Bearer test-key-xyz"
        assert captured["headers"]["content-type"] == "application/json"
        assert captured["body"]["model"] == "embed-multilingual-v3.0"
        assert captured["body"]["texts"] == ["butter chicken", "paneer tikka"]
        assert captured["body"]["embedding_types"] == ["float"]

    def test_extracts_embeddings_and_l2_normalizes(self):
        """Embeddings are pulled from the 'float' nesting, truncated to the
        requested dimension, and L2-normalized to unit length."""

        def urlopen_fn(req, timeout=None):
            body = json.loads(req.data.decode())
            n = len(body["texts"])
            payload = {"embeddings": {"float": [[float(i + 1)] * 8 for i in range(n)]}}
            return BytesIO(json.dumps(payload).encode())

        adapter, patches = _make_cohere_adapter(urlopen_fn, dimension=4)
        try:
            out = adapter.encode(["x", "y", "z"], normalize=True)
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (3, 4)
        assert out.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.ones(3), atol=1e-6)

    def test_batches_at_96(self):
        """Cohere caps each request at 96 texts."""
        batch_sizes: list[int] = []

        def urlopen_fn(req, timeout=None):
            body = json.loads(req.data.decode())
            batch_sizes.append(len(body["texts"]))
            n = len(body["texts"])
            payload = {"embeddings": {"float": [[1.0] * 8 for _ in range(n)]}}
            return BytesIO(json.dumps(payload).encode())

        adapter, patches = _make_cohere_adapter(urlopen_fn, dimension=4)
        try:
            out = adapter.encode([f"t{i}" for i in range(100)])
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (100, 4)
        assert batch_sizes == [96, 4]

    def test_empty_input_returns_empty_array(self):
        """encode([]) returns shape (0, dim) without opening any connection."""
        calls: list = []

        def urlopen_fn(req, timeout=None):  # pragma: no cover - must not run
            calls.append(req)
            raise AssertionError("urlopen should not be called for empty input")

        adapter, patches = _make_cohere_adapter(urlopen_fn, dimension=384)
        try:
            out = adapter.encode([])
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (0, 384)
        assert calls == []

    def test_requested_dimension_wider_than_native_raises(self):
        """A requested dimension wider than what the API returns is rejected
        rather than silently returning a narrower array."""

        def urlopen_fn(req, timeout=None):
            body = json.loads(req.data.decode())
            n = len(body["texts"])
            # Return only 8-wide rows.
            payload = {"embeddings": {"float": [[1.0] * 8 for _ in range(n)]}}
            return BytesIO(json.dumps(payload).encode())

        adapter, patches = _make_cohere_adapter(urlopen_fn, dimension=16)
        try:
            with pytest.raises(ValueError, match="dimension 16 was requested"):
                adapter.encode(["a", "b"])
        finally:
            for p in patches:
                p.stop()


# =========================================================================
# Gemini adapter: construction-only tests
# =========================================================================


class TestGeminiAdapterConstruction:
    """Gemini adapter construction and error handling."""

    def test_missing_api_key_raises(self, monkeypatch):
        """Should raise ValueError when no API key is available."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from foodeval.adapters.gemini_adapter import GeminiAdapter

        with pytest.raises(ValueError, match="Gemini API key"):
            GeminiAdapter("gemini-embedding-2", api_key=None)

    def test_name_format_includes_dimension(self):
        """When dimension is set, the name should include it."""
        from foodeval.adapters.gemini_adapter import GeminiAdapter

        adapter = GeminiAdapter(
            "gemini-embedding-2",
            dimension=384,
            api_key="test-key-for-unit-test",
        )
        assert adapter.name == "gemini-embedding-2-384d"
        assert adapter.dimension == 384

    def test_native_dimension_when_none(self):
        """dimension=None falls back to the native 3072-wide output."""
        from foodeval.adapters.gemini_adapter import GeminiAdapter

        adapter = GeminiAdapter(
            "gemini-embedding-2",
            dimension=None,
            api_key="test-key-for-unit-test",
        )
        assert adapter.dimension == 3072
        assert adapter.name == "gemini-embedding-2"


# =========================================================================
# Gemini adapter: encode() behavior with mocked urlopen
# =========================================================================


def _make_gemini_adapter(urlopen_fn, dimension=4, model="gemini-embedding-2"):
    """Build a GeminiAdapter whose urllib urlopen runs urlopen_fn.

    urlopen_fn(request) returns a file-like object whose .read() yields the
    JSON response bytes. Caching is patched out.
    """
    patches = [
        patch(
            "foodeval.adapters.gemini_adapter.urllib.request.urlopen",
            side_effect=urlopen_fn,
        ),
        patch("foodeval.adapters.gemini_adapter.load_cache", return_value=None),
        patch("foodeval.adapters.gemini_adapter.save_cache"),
    ]
    for p in patches:
        p.start()
    try:
        from foodeval.adapters.gemini_adapter import GeminiAdapter

        adapter = GeminiAdapter(model, dimension=dimension, api_key="test-key-xyz")
    except Exception:
        for p in patches:
            p.stop()
        raise
    return adapter, patches


def _gemini_payload(n: int, width: int) -> bytes:
    """JSON response bytes with n embeddings of the given width, in order."""
    payload = {"embeddings": [{"values": [float(i + 1)] * width} for i in range(n)]}
    return json.dumps(payload).encode()


class TestGeminiAdapterEncode:
    """GeminiAdapter.encode: request shape, headers, parsing, batching, normalization."""

    def test_posts_to_batch_embed_with_key_header_and_body(self):
        """The request hits the batchEmbedContents endpoint with the API key
        header and the expected per-text JSON body (model, content, taskType,
        outputDimensionality)."""
        captured: dict = {}

        def urlopen_fn(req, timeout=None):
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            captured["headers"] = {k.lower(): v for k, v in req.header_items()}
            captured["body"] = json.loads(req.data.decode())
            return BytesIO(_gemini_payload(len(captured["body"]["requests"]), 4))

        adapter, patches = _make_gemini_adapter(urlopen_fn, dimension=4)
        try:
            adapter.encode(["butter chicken", "paneer tikka"])
        finally:
            for p in patches:
                p.stop()
        assert captured["url"] == (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-embedding-2:batchEmbedContents"
        )
        assert captured["method"] == "POST"
        assert captured["headers"]["x-goog-api-key"] == "test-key-xyz"
        assert captured["headers"]["content-type"] == "application/json"
        requests = captured["body"]["requests"]
        assert [r["content"]["parts"][0]["text"] for r in requests] == [
            "butter chicken",
            "paneer tikka",
        ]
        assert all(r["model"] == "models/gemini-embedding-2" for r in requests)
        assert all(r["outputDimensionality"] == 4 for r in requests)

    def test_query_and_document_task_types(self):
        """encode_queries sends RETRIEVAL_QUERY; encode and encode_documents
        send RETRIEVAL_DOCUMENT."""
        task_types: list[str] = []

        def urlopen_fn(req, timeout=None):
            body = json.loads(req.data.decode())
            task_types.extend(item["taskType"] for item in body["requests"])
            return BytesIO(_gemini_payload(len(body["requests"]), 4))

        adapter, patches = _make_gemini_adapter(urlopen_fn, dimension=4)
        try:
            adapter.encode_queries(["butter chicken"])
            adapter.encode_documents(["paneer tikka"])
            adapter.encode(["iced tea"])
        finally:
            for p in patches:
                p.stop()
        assert task_types == [
            "RETRIEVAL_QUERY",
            "RETRIEVAL_DOCUMENT",
            "RETRIEVAL_DOCUMENT",
        ]

    def test_extracts_embeddings_and_l2_normalizes(self):
        """Embeddings are pulled from the 'values' key in request order and
        L2-normalized to unit-length float32."""

        def urlopen_fn(req, timeout=None):
            body = json.loads(req.data.decode())
            return BytesIO(_gemini_payload(len(body["requests"]), 4))

        adapter, patches = _make_gemini_adapter(urlopen_fn, dimension=4)
        try:
            out = adapter.encode(["x", "y", "z"], normalize=True)
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (3, 4)
        assert out.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.ones(3), atol=1e-6)

    def test_batches_at_100(self):
        """Gemini caps each batchEmbedContents request at 100 texts."""
        batch_sizes: list[int] = []

        def urlopen_fn(req, timeout=None):
            body = json.loads(req.data.decode())
            batch_sizes.append(len(body["requests"]))
            return BytesIO(_gemini_payload(len(body["requests"]), 4))

        adapter, patches = _make_gemini_adapter(urlopen_fn, dimension=4)
        try:
            out = adapter.encode([f"t{i}" for i in range(150)])
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (150, 4)
        assert batch_sizes == [100, 50]

    def test_empty_input_returns_empty_array(self):
        """encode([]) returns shape (0, dim) without opening any connection."""
        calls: list = []

        def urlopen_fn(req, timeout=None):  # pragma: no cover - must not run
            calls.append(req)
            raise AssertionError("urlopen should not be called for empty input")

        adapter, patches = _make_gemini_adapter(urlopen_fn, dimension=384)
        try:
            out = adapter.encode([])
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (0, 384)
        assert calls == []

    def test_dimension_mismatch_raises(self):
        """If the API ignores outputDimensionality and returns a different
        width, raise ValueError instead of silently accepting it."""

        def urlopen_fn(req, timeout=None):
            body = json.loads(req.data.decode())
            # Return 8-wide rows while the adapter expects 4.
            return BytesIO(_gemini_payload(len(body["requests"]), 8))

        adapter, patches = _make_gemini_adapter(urlopen_fn, dimension=4)
        try:
            with pytest.raises(ValueError, match="adapter reports dimension 4"):
                adapter.encode(["a", "b"])
        finally:
            for p in patches:
                p.stop()

    def test_4xx_fails_fast_with_body_detail(self):
        """A non-429 4xx raises immediately (no retries) with the response
        body surfaced in the exception."""
        calls: list = []

        def urlopen_fn(req, timeout=None):
            calls.append(req)
            raise urllib.error.HTTPError(
                req.full_url,
                400,
                "Bad Request",
                None,
                BytesIO(b'{"error": {"message": "invalid taskType"}}'),
            )

        adapter, patches = _make_gemini_adapter(urlopen_fn, dimension=4)
        try:
            with pytest.raises(urllib.error.HTTPError, match="invalid taskType"):
                adapter.encode(["a"])
        finally:
            for p in patches:
                p.stop()
        assert len(calls) == 1


# =========================================================================
# Vertex adapter: construction-only tests
# =========================================================================


class TestVertexAdapterConstruction:
    """Vertex adapter construction and error handling."""

    def test_missing_project_raises(self, monkeypatch):
        """Should raise ValueError when no GCP project is available."""
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        from foodeval.adapters.vertex_adapter import VertexAdapter

        with pytest.raises(ValueError, match="Google Cloud project"):
            VertexAdapter("gemini-embedding-001", project=None)

    def test_name_format_includes_dimension_without_vertex_prefix(self):
        """The name is the model name plus dimension, with no vertex prefix,
        so results and the embedding cache stay transport-agnostic."""
        from foodeval.adapters.vertex_adapter import VertexAdapter

        adapter = VertexAdapter(
            "gemini-embedding-001",
            dimension=384,
            project="test-project",
        )
        assert adapter.name == "gemini-embedding-001-384d"
        assert adapter.dimension == 384

    def test_native_dimension_when_none(self):
        """dimension=None falls back to the native 3072-wide output."""
        from foodeval.adapters.vertex_adapter import VertexAdapter

        adapter = VertexAdapter(
            "gemini-embedding-001",
            dimension=None,
            project="test-project",
        )
        assert adapter.dimension == 3072
        assert adapter.name == "gemini-embedding-001"


# =========================================================================
# Vertex adapter: encode() behavior with mocked urlopen and token call
# =========================================================================


def _make_vertex_adapter(urlopen_fn, dimension=4, model="gemini-embedding-001"):
    """Build a VertexAdapter whose urllib urlopen runs urlopen_fn.

    urlopen_fn(request) returns a file-like object whose .read() yields the
    JSON response bytes. The gcloud token subprocess is mocked to return a
    fixed token, and caching is patched out.
    """
    patches = [
        patch(
            "foodeval.adapters.vertex_adapter.urllib.request.urlopen",
            side_effect=urlopen_fn,
        ),
        patch(
            "foodeval.adapters.vertex_adapter.subprocess.run",
            return_value=SimpleNamespace(stdout="test-token-abc\n", returncode=0),
        ),
        patch("foodeval.adapters.vertex_adapter.load_cache", return_value=None),
        patch("foodeval.adapters.vertex_adapter.save_cache"),
    ]
    for p in patches:
        p.start()
    try:
        from foodeval.adapters.vertex_adapter import VertexAdapter

        adapter = VertexAdapter(model, dimension=dimension, project="test-project")
    except Exception:
        for p in patches:
            p.stop()
        raise
    return adapter, patches


def _vertex_payload(n: int, width: int) -> bytes:
    """JSON response bytes with n predictions of the given width, in order."""
    payload = {
        "predictions": [
            {"embeddings": {"values": [float(i + 1)] * width}} for i in range(n)
        ]
    }
    return json.dumps(payload).encode()


class TestVertexAdapterEncode:
    """VertexAdapter.encode: request shape, headers, parsing, batching, normalization."""

    def test_posts_to_predict_with_bearer_header_and_body(self):
        """The request hits the Vertex :predict endpoint (project, location,
        model in the URL) with a Bearer token header and the expected
        instances/parameters JSON body."""
        captured: dict = {}

        def urlopen_fn(req, timeout=None):
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            captured["headers"] = {k.lower(): v for k, v in req.header_items()}
            captured["body"] = json.loads(req.data.decode())
            return BytesIO(_vertex_payload(len(captured["body"]["instances"]), 4))

        adapter, patches = _make_vertex_adapter(urlopen_fn, dimension=4)
        try:
            adapter.encode(["butter chicken", "paneer tikka"])
        finally:
            for p in patches:
                p.stop()
        assert captured["url"] == (
            "https://aiplatform.googleapis.com/v1/projects/test-project/"
            "locations/global/publishers/google/models/"
            "gemini-embedding-001:predict"
        )
        assert captured["method"] == "POST"
        assert captured["headers"]["authorization"] == "Bearer test-token-abc"
        assert captured["headers"]["content-type"] == "application/json"
        instances = captured["body"]["instances"]
        assert [item["content"] for item in instances] == [
            "butter chicken",
            "paneer tikka",
        ]
        assert all(item["task_type"] == "RETRIEVAL_DOCUMENT" for item in instances)
        assert captured["body"]["parameters"] == {"outputDimensionality": 4}

    def test_query_and_document_task_types(self):
        """encode_queries sends RETRIEVAL_QUERY; encode and encode_documents
        send RETRIEVAL_DOCUMENT."""
        task_types: list[str] = []

        def urlopen_fn(req, timeout=None):
            body = json.loads(req.data.decode())
            task_types.extend(item["task_type"] for item in body["instances"])
            return BytesIO(_vertex_payload(len(body["instances"]), 4))

        adapter, patches = _make_vertex_adapter(urlopen_fn, dimension=4)
        try:
            adapter.encode_queries(["butter chicken"])
            adapter.encode_documents(["paneer tikka"])
            adapter.encode(["iced tea"])
        finally:
            for p in patches:
                p.stop()
        assert task_types == [
            "RETRIEVAL_QUERY",
            "RETRIEVAL_DOCUMENT",
            "RETRIEVAL_DOCUMENT",
        ]

    def test_extracts_embeddings_and_l2_normalizes(self):
        """Embeddings are pulled from predictions[*].embeddings.values in
        request order and L2-normalized to unit-length float32."""

        def urlopen_fn(req, timeout=None):
            body = json.loads(req.data.decode())
            return BytesIO(_vertex_payload(len(body["instances"]), 4))

        adapter, patches = _make_vertex_adapter(urlopen_fn, dimension=4)
        try:
            out = adapter.encode(["x", "y", "z"], normalize=True)
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (3, 4)
        assert out.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.ones(3), atol=1e-6)

    def test_native_dimension_omits_output_dimensionality(self):
        """dimension=None requests should not carry a parameters block."""
        bodies: list[dict] = []

        def urlopen_fn(req, timeout=None):
            body = json.loads(req.data.decode())
            bodies.append(body)
            return BytesIO(_vertex_payload(len(body["instances"]), 3072))

        adapter, patches = _make_vertex_adapter(urlopen_fn, dimension=None)
        try:
            out = adapter.encode(["a"])
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (1, 3072)
        assert "parameters" not in bodies[0]

    def test_batches_at_250(self):
        """Vertex caps each :predict request at 250 instances."""
        batch_sizes: list[int] = []

        def urlopen_fn(req, timeout=None):
            body = json.loads(req.data.decode())
            batch_sizes.append(len(body["instances"]))
            return BytesIO(_vertex_payload(len(body["instances"]), 4))

        adapter, patches = _make_vertex_adapter(urlopen_fn, dimension=4)
        try:
            out = adapter.encode([f"t{i}" for i in range(300)])
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (300, 4)
        assert batch_sizes == [250, 50]

    def test_empty_input_returns_empty_array(self):
        """encode([]) returns shape (0, dim) without opening any connection."""
        calls: list = []

        def urlopen_fn(req, timeout=None):  # pragma: no cover - must not run
            calls.append(req)
            raise AssertionError("urlopen should not be called for empty input")

        adapter, patches = _make_vertex_adapter(urlopen_fn, dimension=384)
        try:
            out = adapter.encode([])
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (0, 384)
        assert calls == []

    def test_dimension_mismatch_raises(self):
        """If the API ignores outputDimensionality and returns a different
        width, raise ValueError instead of silently accepting it."""

        def urlopen_fn(req, timeout=None):
            body = json.loads(req.data.decode())
            # Return 8-wide rows while the adapter expects 4.
            return BytesIO(_vertex_payload(len(body["instances"]), 8))

        adapter, patches = _make_vertex_adapter(urlopen_fn, dimension=4)
        try:
            with pytest.raises(ValueError, match="adapter reports dimension 4"):
                adapter.encode(["a", "b"])
        finally:
            for p in patches:
                p.stop()

    def test_4xx_fails_fast_with_body_detail(self):
        """A non-429 4xx raises immediately (no retries) with the response
        body surfaced in the exception."""
        calls: list = []

        def urlopen_fn(req, timeout=None):
            calls.append(req)
            raise urllib.error.HTTPError(
                req.full_url,
                400,
                "Bad Request",
                None,
                BytesIO(b'{"error": {"message": "invalid task_type"}}'),
            )

        adapter, patches = _make_vertex_adapter(urlopen_fn, dimension=4)
        try:
            with pytest.raises(urllib.error.HTTPError, match="invalid task_type"):
                adapter.encode(["a"])
        finally:
            for p in patches:
                p.stop()
        assert len(calls) == 1

    def test_401_refreshes_token_once_and_retries(self):
        """A 401 invalidates the cached token, refreshes via gcloud, and
        retries the request without consuming a retry attempt."""
        calls: list = []

        def urlopen_fn(req, timeout=None):
            calls.append(req)
            if len(calls) == 1:
                raise urllib.error.HTTPError(
                    req.full_url,
                    401,
                    "Unauthorized",
                    None,
                    BytesIO(b'{"error": {"message": "token expired"}}'),
                )
            body = json.loads(req.data.decode())
            return BytesIO(_vertex_payload(len(body["instances"]), 4))

        adapter, patches = _make_vertex_adapter(urlopen_fn, dimension=4)
        try:
            from foodeval.adapters import vertex_adapter

            out = adapter.encode(["a"])
            # Token fetched once up front, then once more after the 401.
            assert vertex_adapter.subprocess.run.call_count == 2
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (1, 4)
        assert len(calls) == 2


# =========================================================================
# Vertex adapter: embedContent method (gemini-embedding-2)
# =========================================================================


def _vertex_embed_content_payload(value: float, width: int) -> bytes:
    """JSON response bytes for one :embedContent request."""
    return json.dumps({"embedding": {"values": [value] * width}}).encode()


class TestVertexAdapterApiMethodResolution:
    """api_method resolution from the model name, with explicit override."""

    def test_gemini_embedding_2_resolves_to_embed_content(self):
        """gemini-embedding-2 is not served via :predict on Vertex, so it
        auto-resolves to the embedContent method."""
        from foodeval.adapters.vertex_adapter import VertexAdapter

        adapter = VertexAdapter(
            "gemini-embedding-2",
            dimension=384,
            project="test-project",
        )
        assert adapter._api_method == "embedContent"
        assert adapter._url.endswith("gemini-embedding-2:embedContent")

    def test_gemini_embedding_001_resolves_to_predict(self):
        """Other publisher models keep the batched :predict method."""
        from foodeval.adapters.vertex_adapter import VertexAdapter

        adapter = VertexAdapter(
            "gemini-embedding-001",
            dimension=384,
            project="test-project",
        )
        assert adapter._api_method == "predict"
        assert adapter._url.endswith("gemini-embedding-001:predict")

    def test_explicit_api_method_overrides_auto_resolution(self):
        """An explicit api_method wins over the model-name heuristic."""
        from foodeval.adapters.vertex_adapter import VertexAdapter

        forced_predict = VertexAdapter(
            "gemini-embedding-2",
            dimension=384,
            project="test-project",
            api_method="predict",
        )
        assert forced_predict._api_method == "predict"
        assert forced_predict._url.endswith("gemini-embedding-2:predict")

        forced_embed = VertexAdapter(
            "gemini-embedding-001",
            dimension=384,
            project="test-project",
            api_method="embedContent",
        )
        assert forced_embed._api_method == "embedContent"
        assert forced_embed._url.endswith("gemini-embedding-001:embedContent")


class TestVertexAdapterEmbedContent:
    """VertexAdapter via :embedContent: one text per request, threaded
    fan-out, input-order results, same retry/auth/cache treatment."""

    def test_posts_single_text_to_embed_content_endpoint(self):
        """Each request hits :embedContent with a Bearer header and the
        content.parts/taskType/outputDimensionality body shape."""
        captured: dict = {}

        def urlopen_fn(req, timeout=None):
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            captured["headers"] = {k.lower(): v for k, v in req.header_items()}
            captured["body"] = json.loads(req.data.decode())
            return BytesIO(_vertex_embed_content_payload(1.0, 4))

        adapter, patches = _make_vertex_adapter(
            urlopen_fn, dimension=4, model="gemini-embedding-2"
        )
        try:
            adapter.encode(["butter chicken"])
        finally:
            for p in patches:
                p.stop()
        assert captured["url"] == (
            "https://aiplatform.googleapis.com/v1/projects/test-project/"
            "locations/global/publishers/google/models/"
            "gemini-embedding-2:embedContent"
        )
        assert captured["method"] == "POST"
        assert captured["headers"]["authorization"] == "Bearer test-token-abc"
        assert captured["body"] == {
            "content": {"parts": [{"text": "butter chicken"}]},
            "taskType": "RETRIEVAL_DOCUMENT",
            "outputDimensionality": 4,
        }

    def test_results_preserve_input_order(self):
        """Threaded fan-out returns embeddings in input order even when
        earlier requests complete last."""

        def urlopen_fn(req, timeout=None):
            body = json.loads(req.data.decode())
            i = int(body["content"]["parts"][0]["text"].removeprefix("t"))
            # Earlier texts finish later, so completion order is reversed.
            time.sleep((9 - i) * 0.005)
            return BytesIO(_vertex_embed_content_payload(float(i + 1), 4))

        adapter, patches = _make_vertex_adapter(
            urlopen_fn, dimension=4, model="gemini-embedding-2"
        )
        try:
            out = adapter.encode([f"t{i}" for i in range(10)], normalize=False)
        finally:
            for p in patches:
                p.stop()
        assert out.shape == (10, 4)
        np.testing.assert_array_equal(out[:, 0], np.arange(1.0, 11.0, dtype=np.float32))

    def test_query_and_document_task_types(self):
        """encode_queries sends RETRIEVAL_QUERY; encode and encode_documents
        send RETRIEVAL_DOCUMENT."""
        task_types: list[str] = []

        def urlopen_fn(req, timeout=None):
            body = json.loads(req.data.decode())
            task_types.append(body["taskType"])
            return BytesIO(_vertex_embed_content_payload(1.0, 4))

        adapter, patches = _make_vertex_adapter(
            urlopen_fn, dimension=4, model="gemini-embedding-2"
        )
        try:
            adapter.encode_queries(["butter chicken"])
            adapter.encode_documents(["paneer tikka"])
            adapter.encode(["iced tea"])
        finally:
            for p in patches:
                p.stop()
        assert task_types == [
            "RETRIEVAL_QUERY",
            "RETRIEVAL_DOCUMENT",
            "RETRIEVAL_DOCUMENT",
        ]

    def test_4xx_fails_fast_with_body_detail(self):
        """A non-429 4xx raises immediately (no retries) with the response
        body surfaced in the exception."""
        calls: list = []

        def urlopen_fn(req, timeout=None):
            calls.append(req)
            raise urllib.error.HTTPError(
                req.full_url,
                400,
                "Bad Request",
                None,
                BytesIO(b'{"error": {"message": "invalid taskType"}}'),
            )

        adapter, patches = _make_vertex_adapter(
            urlopen_fn, dimension=4, model="gemini-embedding-2"
        )
        try:
            with pytest.raises(urllib.error.HTTPError, match="invalid taskType"):
                adapter.encode(["a"])
        finally:
            for p in patches:
                p.stop()
        assert len(calls) == 1


# =========================================================================
# Factory functions in adapters/__init__.py
# =========================================================================


class TestAdapterFactories:
    """Factory functions for lazy adapter creation."""

    def test_get_bm25_adapter_returns_bm25(self):
        from foodeval.adapters import get_bm25_adapter

        adapter = get_bm25_adapter()
        assert adapter.name == "Lexical (TF)"

    def test_get_openai_adapter_raises_without_key(self, monkeypatch):
        from foodeval.adapters import get_openai_adapter

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises((ValueError, ImportError)):
            get_openai_adapter(api_key=None)


# =========================================================================
# Cache helpers
# =========================================================================


class TestCacheHelpers:
    """Disk cache for embedding results."""

    def test_cache_key_deterministic(self):
        """Same inputs produce the same cache key."""
        k1 = cache_key("model-a", 384, ["butter chicken", "iced tea"])
        k2 = cache_key("model-a", 384, ["butter chicken", "iced tea"])
        assert k1 == k2

    def test_cache_key_order_dependent(self):
        """Cache key should differ when text order changes (embeddings are positional)."""
        k1 = cache_key("model-a", 384, ["butter chicken", "iced tea"])
        k2 = cache_key("model-a", 384, ["iced tea", "butter chicken"])
        assert k1 != k2

    def test_cache_key_differs_by_model(self):
        k1 = cache_key("model-a", 384, ["butter chicken"])
        k2 = cache_key("model-b", 384, ["butter chicken"])
        assert k1 != k2

    def test_cache_key_differs_by_dimension(self):
        k1 = cache_key("model-a", 384, ["butter chicken"])
        k2 = cache_key("model-a", 768, ["butter chicken"])
        assert k1 != k2

    def test_cache_key_differs_by_content(self):
        k1 = cache_key("model-a", 384, ["butter chicken"])
        k2 = cache_key("model-a", 384, ["paneer tikka"])
        assert k1 != k2

    def test_cache_key_is_hex_string(self):
        k = cache_key("model-a", 384, ["butter chicken"])
        assert len(k) == 64  # SHA-256
        assert all(c in "0123456789abcdef" for c in k)

    def test_save_and_load_roundtrip(self, tmp_path):
        """Saved embeddings should be recoverable via load_cache."""
        with patch("foodeval.adapters.base.CACHE_DIR", tmp_path):
            texts = ["butter chicken", "paneer tikka"]
            embeddings = np.random.randn(2, 384).astype(np.float32)
            save_cache("test-model", 384, texts, embeddings)
            loaded = load_cache("test-model", 384, texts)
            assert loaded is not None
            np.testing.assert_array_equal(loaded, embeddings)

    def test_load_cache_returns_none_for_missing(self, tmp_path):
        with patch("foodeval.adapters.base.CACHE_DIR", tmp_path):
            loaded = load_cache("nonexistent-model", 384, ["butter chicken"])
            assert loaded is None

    def test_load_cache_rejects_wrong_length(self, tmp_path):
        """If the cached array has a different row count, reject it."""
        with patch("foodeval.adapters.base.CACHE_DIR", tmp_path):
            texts = ["butter chicken"]
            # Save with 2 rows but only 1 text
            embeddings = np.random.randn(2, 384).astype(np.float32)
            save_cache("test-model", 384, texts, embeddings)
            loaded = load_cache("test-model", 384, texts)
            assert loaded is None

    def test_save_creates_cache_directory(self, tmp_path):
        """save_cache should create the cache directory if it does not exist."""
        cache_dir = tmp_path / "nonexistent_subdir"
        with patch("foodeval.adapters.base.CACHE_DIR", cache_dir):
            texts = ["dal makhani"]
            embeddings = np.random.randn(1, 64).astype(np.float32)
            save_cache("test-model", 64, texts, embeddings)
            assert cache_dir.exists()

    def test_cache_roundtrip_preserves_dtype(self, tmp_path):
        """Cached embeddings should be float32 after loading."""
        with patch("foodeval.adapters.base.CACHE_DIR", tmp_path):
            texts = ["green curry"]
            embeddings = np.random.randn(1, 128).astype(np.float64)
            save_cache("test-model", 128, texts, embeddings)
            loaded = load_cache("test-model", 128, texts)
            assert loaded is not None
            assert loaded.dtype == np.float32
