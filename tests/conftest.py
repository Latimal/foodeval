"""Shared fixtures for FoodEval test suite.

Provides deterministic mock adapters, minimal data dicts for each task type,
and helpers that write temporary data files for task loading tests.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mock embedding adapter: deterministic random embeddings
# ---------------------------------------------------------------------------


class DummyAdapter:
    """Deterministic mock adapter that produces stable random embeddings.

    Each text's embedding is seeded from a stable hash of its content
    (``zlib.crc32``), so identical input texts always produce identical
    vectors -- not just within one run, but across separate processes and
    adapter instances. This makes cosine-similarity, ranking, and reproducibility
    assertions robust to Python's per-process hash randomization (PYTHONHASHSEED).
    """

    def __init__(self, dim: int = 64, seed: int = 12345) -> None:
        self._dim = dim
        self._seed = seed

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        # Seed each text from a stable hash of its content so the embedding
        # depends only on the text, not on call order or the process's
        # PYTHONHASHSEED. The built-in hash() is salted per process and would
        # make embeddings differ across runs, breaking reproducibility.
        embeddings = np.empty((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            text_seed = zlib.crc32(text.encode("utf-8"))
            text_rng = np.random.RandomState(text_seed)
            embeddings[i] = text_rng.randn(self._dim).astype(np.float32)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            embeddings = embeddings / norms
        return embeddings

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"dummy-{self._dim}d"


class ConstantAdapter:
    """Adapter that returns the same fixed vector for every input.

    Useful for testing edge cases where all embeddings are identical, such as
    verifying that dedup tasks degrade gracefully when the model has zero
    discriminative power.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self._vector = np.ones(dim, dtype=np.float32)
        self._vector /= np.linalg.norm(self._vector)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        embeddings = np.tile(self._vector, (len(texts), 1))
        return embeddings

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"constant-{self._dim}d"


@pytest.fixture(autouse=True)
def reset_task_registry_cache():
    """Clear each registry task's cached data before and after every test.

    TASK_REGISTRY holds module-level singleton task instances. Once a task's
    ``run``/``describe`` loads data, it caches it on ``_data`` (and the parsed
    pairs/corpus/items) for the process lifetime. Without this reset, a test
    that loads real benchmark data would leave that data cached for every
    later test that touches the same singleton, so test outcomes could depend
    on execution order. Resetting ``_data`` to None forces a fresh load and
    keeps the shared singletons isolated between tests.
    """
    from foodeval.tasks import TASK_REGISTRY

    def _clear() -> None:
        for task in TASK_REGISTRY.values():
            task._data = None
            # Subclasses cache the parsed payload separately; clear whichever
            # the task type exposes so a stale load can't leak across tests.
            for attr in ("_pairs", "_corpus", "_queries", "_items"):
                if hasattr(task, attr):
                    setattr(task, attr, [])

    _clear()
    yield
    _clear()


@pytest.fixture
def dummy_adapter() -> DummyAdapter:
    """A deterministic mock adapter with 64-dimensional embeddings."""
    return DummyAdapter(dim=64, seed=12345)


@pytest.fixture
def constant_adapter() -> ConstantAdapter:
    """An adapter that returns identical embeddings for all inputs."""
    return ConstantAdapter(dim=64)


# ---------------------------------------------------------------------------
# Minimal data dicts for each task type
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_retrieval_data() -> dict[str, Any]:
    """Minimal retrieval task data with 4 corpus items, 2 queries."""
    return {
        "task": "test_search",
        "version": "0.1.0",
        "description": "Test search task",
        "metric": "ndcg@10",
        "corpus": [
            "butter chicken",
            "paneer tikka",
            "iced tea",
            "mango lassi",
        ],
        "queries": [
            {
                "id": "q001",
                "query": "indian chicken curry",
                "domain": "indian",
                "relevance": {
                    "butter chicken": 3,
                    "paneer tikka": 1,
                },
            },
            {
                "id": "q002",
                "query": "cold beverage",
                "domain": "beverage",
                "relevance": {
                    "iced tea": 3,
                    "mango lassi": 2,
                },
            },
        ],
        "metadata": {
            "n_queries": 2,
            "n_corpus": 4,
            "domains": ["indian", "beverage"],
        },
    }


@pytest.fixture
def sample_dedup_data() -> dict[str, Any]:
    """Minimal pair classification data with 6 pairs (3 positive, 3 negative)."""
    return {
        "task": "test_dedup",
        "version": "0.1.0",
        "description": "Test dedup task",
        "metric": "best_f1",
        "pairs": [
            {
                "id": "p001",
                "text_a": "Butter Chicken",
                "text_b": "Murgh Makhani",
                "label": 1,
                "domain": "indian",
            },
            {
                "id": "p002",
                "text_a": "Chicken Tikka",
                "text_b": "Chicken Tikka Masala",
                "label": 0,
                "domain": "indian",
            },
            {
                "id": "p003",
                "text_a": "Iced Latte",
                "text_b": "Cold Latte",
                "label": 1,
                "domain": "beverage",
            },
            {
                "id": "p004",
                "text_a": "Green Tea",
                "text_b": "Matcha Latte",
                "label": 0,
                "domain": "beverage",
            },
            {
                "id": "p005",
                "text_a": "Margherita Pizza",
                "text_b": "Margherita",
                "label": 1,
                "domain": "global",
            },
            {
                "id": "p006",
                "text_a": "Fish Tacos",
                "text_b": "Chicken Tacos",
                "label": 0,
                "domain": "global",
            },
        ],
        "metadata": {
            "n_pairs": 6,
            "n_positive": 3,
            "n_negative": 3,
            "domains": ["indian", "beverage", "global"],
        },
    }


@pytest.fixture
def sample_classification_data() -> dict[str, Any]:
    """Minimal classification data with 3 classes, 12 items."""
    return {
        "task": "test_classify",
        "version": "0.1.0",
        "description": "Test classification task",
        "metric": "macro_f1",
        "items": [
            {
                "id": "i001",
                "text": "butter chicken",
                "label": "Indian",
                "source": "test",
            },
            {"id": "i002", "text": "paneer tikka", "label": "Indian", "source": "test"},
            {"id": "i003", "text": "dal makhani", "label": "Indian", "source": "test"},
            {
                "id": "i004",
                "text": "chicken biryani",
                "label": "Indian",
                "source": "test",
            },
            {
                "id": "i005",
                "text": "margherita pizza",
                "label": "Italian",
                "source": "test",
            },
            {
                "id": "i006",
                "text": "spaghetti carbonara",
                "label": "Italian",
                "source": "test",
            },
            {"id": "i007", "text": "risotto", "label": "Italian", "source": "test"},
            {
                "id": "i008",
                "text": "penne arrabbiata",
                "label": "Italian",
                "source": "test",
            },
            {"id": "i009", "text": "pad thai", "label": "Thai", "source": "test"},
            {"id": "i010", "text": "green curry", "label": "Thai", "source": "test"},
            {"id": "i011", "text": "tom yum soup", "label": "Thai", "source": "test"},
            {"id": "i012", "text": "massaman curry", "label": "Thai", "source": "test"},
        ],
        "label_names": ["Indian", "Italian", "Thai"],
        "metadata": {
            "n_items": 12,
            "n_classes": 3,
        },
    }


# ---------------------------------------------------------------------------
# Temp data file helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def write_task_json(tmp_path: Path):
    """Factory fixture that writes a data dict to a JSON file in tmp_path.

    Returns a callable: write_task_json(name, data) -> Path
    """

    def _write(name: str, data: dict[str, Any]) -> Path:
        fpath = tmp_path / f"{name}.json"
        fpath.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return fpath

    return _write
