"""Tests for task loading, validation, and evaluation.

Uses temporary data files and the DummyAdapter from conftest to test task
behavior without requiring real models or the production data files.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from foodeval.tasks.base import BenchmarkTask, TaskResult, DATA_DIR
from foodeval.tasks.retrieval import RetrievalTask
from foodeval.tasks.pair_classification import PairClassificationTask
from foodeval.tasks.classification import ClassificationTask


# =========================================================================
# TaskResult
# =========================================================================


class TestTaskResult:
    """TaskResult dataclass serialization."""

    def test_to_dict_returns_all_fields(self):
        result = TaskResult(
            task_name="food_search",
            main_score=0.8765,
            metric_name="ndcg@10",
            details={"k": 10},
            n_examples=50,
            elapsed_seconds=1.234,
        )
        d = result.to_dict()
        assert d["task_name"] == "food_search"
        assert d["main_score"] == 0.8765
        assert d["metric_name"] == "ndcg@10"
        assert d["details"] == {"k": 10}
        assert d["n_examples"] == 50
        assert d["elapsed_seconds"] == 1.234

    def test_to_dict_rounds_elapsed(self):
        result = TaskResult(
            task_name="test",
            main_score=0.5,
            metric_name="f1",
            elapsed_seconds=1.23456789,
        )
        assert result.to_dict()["elapsed_seconds"] == 1.235

    def test_default_values(self):
        result = TaskResult(task_name="test", main_score=0.5, metric_name="f1")
        assert result.details == {}
        assert result.n_examples == 0
        assert result.elapsed_seconds == 0.0

    def test_to_dict_is_json_serializable(self):
        result = TaskResult(
            task_name="test",
            main_score=0.85,
            metric_name="ndcg@10",
            details={"nested": {"key": [1, 2, 3]}},
        )
        # Should not raise
        serialized = json.dumps(result.to_dict())
        assert isinstance(serialized, str)


# =========================================================================
# Retrieval task
# =========================================================================


class TestRetrievalTask:
    """RetrievalTask loading, validation, and evaluation."""

    def test_load_valid_data(self, tmp_path, sample_retrieval_data, dummy_adapter):
        """Should load without error from a well-formed JSON file."""
        data_file = tmp_path / "test_search.json"
        data_file.write_text(json.dumps(sample_retrieval_data), encoding="utf-8")

        task = RetrievalTask("test_search")
        task._data_path = data_file
        task.load_data()
        assert len(task._corpus) == 4
        assert len(task._queries) == 2

    def test_missing_data_file_raises(self, tmp_path):
        task = RetrievalTask("nonexistent_task")
        task._data_path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            task.load_data()

    def test_empty_corpus_raises(self, tmp_path):
        data = {
            "task": "empty_search",
            "version": "0.1.0",
            "corpus": [],
            "queries": [{"id": "q1", "query": "test", "relevance": {}}],
        }
        data_file = tmp_path / "empty_search.json"
        data_file.write_text(json.dumps(data), encoding="utf-8")

        task = RetrievalTask("empty_search")
        task._data_path = data_file
        with pytest.raises(ValueError, match="corpus is empty"):
            task.load_data()

    def test_empty_queries_raises(self, tmp_path):
        data = {
            "task": "no_queries",
            "version": "0.1.0",
            "corpus": ["butter chicken"],
            "queries": [],
        }
        data_file = tmp_path / "no_queries.json"
        data_file.write_text(json.dumps(data), encoding="utf-8")

        task = RetrievalTask("no_queries")
        task._data_path = data_file
        with pytest.raises(ValueError, match="no queries found"):
            task.load_data()

    def test_relevance_referencing_missing_corpus_item_raises(self, tmp_path):
        data = {
            "task": "bad_ref",
            "version": "0.1.0",
            "corpus": ["butter chicken"],
            "queries": [
                {
                    "id": "q1",
                    "query": "test",
                    "relevance": {"nonexistent item": 3},
                }
            ],
        }
        data_file = tmp_path / "bad_ref.json"
        data_file.write_text(json.dumps(data), encoding="utf-8")

        task = RetrievalTask("bad_ref")
        task._data_path = data_file
        with pytest.raises(ValueError, match="not in corpus"):
            task.load_data()

    def test_evaluate_returns_task_result(
        self, tmp_path, sample_retrieval_data, dummy_adapter
    ):
        data_file = tmp_path / "test_search.json"
        data_file.write_text(json.dumps(sample_retrieval_data), encoding="utf-8")

        task = RetrievalTask("test_search")
        task._data_path = data_file
        task.load_data()
        result = task.evaluate(dummy_adapter)

        assert isinstance(result, TaskResult)
        assert result.task_name == "test_search"
        assert result.metric_name == "ndcg@10"
        assert 0.0 <= result.main_score <= 1.0
        assert result.n_examples == 2

    def test_evaluate_details_structure(
        self, tmp_path, sample_retrieval_data, dummy_adapter
    ):
        data_file = tmp_path / "test_search.json"
        data_file.write_text(json.dumps(sample_retrieval_data), encoding="utf-8")

        task = RetrievalTask("test_search")
        task._data_path = data_file
        task.load_data()
        result = task.evaluate(dummy_adapter)

        assert "k" in result.details
        assert "n_corpus" in result.details
        assert "per_domain" in result.details
        assert "confidence_interval" in result.details
        assert "per_query" in result.details

    def test_evaluate_per_domain_breakdown(
        self, tmp_path, sample_retrieval_data, dummy_adapter
    ):
        data_file = tmp_path / "test_search.json"
        data_file.write_text(json.dumps(sample_retrieval_data), encoding="utf-8")

        task = RetrievalTask("test_search")
        task._data_path = data_file
        task.load_data()
        result = task.evaluate(dummy_adapter)

        per_domain = result.details["per_domain"]
        assert "indian" in per_domain
        assert "beverage" in per_domain
        for domain_info in per_domain.values():
            assert "mean_ndcg" in domain_info
            assert "n_queries" in domain_info

    def test_run_measures_time(self, tmp_path, sample_retrieval_data, dummy_adapter):
        data_file = tmp_path / "test_search.json"
        data_file.write_text(json.dumps(sample_retrieval_data), encoding="utf-8")

        task = RetrievalTask("test_search")
        task._data_path = data_file
        result = task.run(dummy_adapter)
        assert result.elapsed_seconds > 0.0

    def test_describe_returns_metadata(self, tmp_path, sample_retrieval_data):
        data_file = tmp_path / "test_search.json"
        data_file.write_text(json.dumps(sample_retrieval_data), encoding="utf-8")

        task = RetrievalTask("test_search")
        task._data_path = data_file
        info = task.describe()
        assert info["name"] == "test_search"
        assert info["task_type"] == "retrieval"
        assert info["metric"] == "ndcg@10"

    def test_custom_k(self, tmp_path, sample_retrieval_data, dummy_adapter):
        data_file = tmp_path / "test_search.json"
        data_file.write_text(json.dumps(sample_retrieval_data), encoding="utf-8")

        task = RetrievalTask("test_search", k=2)
        task._data_path = data_file
        task.load_data()
        result = task.evaluate(dummy_adapter)
        assert result.details["k"] == 2

    def test_evaluate_without_load_data_raises(self, dummy_adapter):
        """Calling evaluate before load_data should raise."""
        task = RetrievalTask("unloaded_task")
        with pytest.raises(RuntimeError, match="load_data"):
            task.evaluate(dummy_adapter)

    def test_per_query_details_has_one_entry_per_query(
        self, tmp_path, sample_retrieval_data, dummy_adapter
    ):
        data_file = tmp_path / "test_search.json"
        data_file.write_text(json.dumps(sample_retrieval_data), encoding="utf-8")

        task = RetrievalTask("test_search")
        task._data_path = data_file
        task.load_data()
        result = task.evaluate(dummy_adapter)

        assert len(result.details["per_query"]) == 2
        for pq in result.details["per_query"]:
            assert "id" in pq
            assert "ndcg" in pq
            assert "domain" in pq

    def test_constant_adapter_still_produces_valid_scores(
        self, tmp_path, sample_retrieval_data, constant_adapter
    ):
        """Even with zero discriminative power, scores should be in [0,1]."""
        data_file = tmp_path / "test_search.json"
        data_file.write_text(json.dumps(sample_retrieval_data), encoding="utf-8")

        task = RetrievalTask("test_search")
        task._data_path = data_file
        task.load_data()
        result = task.evaluate(constant_adapter)

        assert 0.0 <= result.main_score <= 1.0

    def test_repr(self):
        task = RetrievalTask("food_search")
        assert "food_search" in repr(task)
        assert "RetrievalTask" in repr(task)


# =========================================================================
# Pair classification task
# =========================================================================


class TestPairClassificationTask:
    """PairClassificationTask loading, validation, and evaluation."""

    def test_load_valid_data(self, tmp_path, sample_dedup_data):
        data_file = tmp_path / "test_dedup.json"
        data_file.write_text(json.dumps(sample_dedup_data), encoding="utf-8")

        task = PairClassificationTask("test_dedup")
        task._data_path = data_file
        task.load_data()
        assert len(task._pairs) == 6

    def test_missing_text_a_raises(self, tmp_path):
        data = {
            "task": "bad_pairs",
            "version": "0.1.0",
            "pairs": [{"id": "p001", "text_b": "Butter Chicken", "label": 1}],
        }
        data_file = tmp_path / "bad_pairs.json"
        data_file.write_text(json.dumps(data), encoding="utf-8")

        task = PairClassificationTask("bad_pairs")
        task._data_path = data_file
        with pytest.raises(ValueError, match="missing text_a or text_b"):
            task.load_data()

    def test_missing_text_b_raises(self, tmp_path):
        data = {
            "task": "bad_pairs",
            "version": "0.1.0",
            "pairs": [{"id": "p001", "text_a": "Butter Chicken", "label": 1}],
        }
        data_file = tmp_path / "bad_pairs.json"
        data_file.write_text(json.dumps(data), encoding="utf-8")

        task = PairClassificationTask("bad_pairs")
        task._data_path = data_file
        with pytest.raises(ValueError, match="missing text_a or text_b"):
            task.load_data()

    def test_invalid_label_raises(self, tmp_path):
        data = {
            "task": "bad_labels",
            "version": "0.1.0",
            "pairs": [{"id": "p001", "text_a": "A", "text_b": "B", "label": 2}],
        }
        data_file = tmp_path / "bad_labels.json"
        data_file.write_text(json.dumps(data), encoding="utf-8")

        task = PairClassificationTask("bad_labels")
        task._data_path = data_file
        with pytest.raises(ValueError, match="invalid label"):
            task.load_data()

    def test_null_label_raises(self, tmp_path):
        data = {
            "task": "null_label",
            "version": "0.1.0",
            "pairs": [{"id": "p001", "text_a": "A", "text_b": "B", "label": None}],
        }
        data_file = tmp_path / "null_label.json"
        data_file.write_text(json.dumps(data), encoding="utf-8")

        task = PairClassificationTask("null_label")
        task._data_path = data_file
        with pytest.raises(ValueError, match="invalid label"):
            task.load_data()

    def test_empty_pairs_raises(self, tmp_path):
        data = {"task": "empty", "version": "0.1.0", "pairs": []}
        data_file = tmp_path / "empty.json"
        data_file.write_text(json.dumps(data), encoding="utf-8")

        task = PairClassificationTask("empty")
        task._data_path = data_file
        with pytest.raises(ValueError, match="no pairs found"):
            task.load_data()

    def test_evaluate_returns_task_result(
        self, tmp_path, sample_dedup_data, dummy_adapter
    ):
        data_file = tmp_path / "test_dedup.json"
        data_file.write_text(json.dumps(sample_dedup_data), encoding="utf-8")

        task = PairClassificationTask("test_dedup")
        task._data_path = data_file
        task.load_data()
        result = task.evaluate(dummy_adapter)

        assert isinstance(result, TaskResult)
        assert result.task_name == "test_dedup"
        assert result.metric_name == "best_f1"
        assert 0.0 <= result.main_score <= 1.0
        assert result.n_examples == 6

    def test_evaluate_details_structure(
        self, tmp_path, sample_dedup_data, dummy_adapter
    ):
        data_file = tmp_path / "test_dedup.json"
        data_file.write_text(json.dumps(sample_dedup_data), encoding="utf-8")

        task = PairClassificationTask("test_dedup")
        task._data_path = data_file
        task.load_data()
        result = task.evaluate(dummy_adapter)

        expected_keys = {
            "best_threshold",
            "precision",
            "recall",
            "average_precision",
            "tp",
            "fp",
            "fn",
            "tn",
            "per_domain",
            "confidence_interval",
        }
        assert expected_keys.issubset(set(result.details.keys()))

    def test_evaluate_per_domain_breakdown(
        self, tmp_path, sample_dedup_data, dummy_adapter
    ):
        data_file = tmp_path / "test_dedup.json"
        data_file.write_text(json.dumps(sample_dedup_data), encoding="utf-8")

        task = PairClassificationTask("test_dedup")
        task._data_path = data_file
        task.load_data()
        result = task.evaluate(dummy_adapter)

        per_domain = result.details["per_domain"]
        assert "indian" in per_domain
        assert "beverage" in per_domain
        assert "global" in per_domain

    def test_confusion_counts_sum(self, tmp_path, sample_dedup_data, dummy_adapter):
        """tp + fp + fn + tn should equal total pairs."""
        data_file = tmp_path / "test_dedup.json"
        data_file.write_text(json.dumps(sample_dedup_data), encoding="utf-8")

        task = PairClassificationTask("test_dedup")
        task._data_path = data_file
        task.load_data()
        result = task.evaluate(dummy_adapter)

        d = result.details
        total = d["tp"] + d["fp"] + d["fn"] + d["tn"]
        assert total == 6

    def test_evaluate_without_load_data_raises(self, dummy_adapter):
        task = PairClassificationTask("unloaded_task")
        with pytest.raises(RuntimeError, match="load_data"):
            task.evaluate(dummy_adapter)

    def test_negative_cosine_pairs_separate_through_task(self, tmp_path):
        """A task whose pairs only separate at a negative threshold still scores.

        Cosine similarity spans [-1, 1]. Here positive pairs sit at cos = -0.2
        and negative pairs at cos = -0.8, so the optimal decision boundary is
        a NEGATIVE threshold. The pair task's threshold sweep must explore that
        range and recover F1 = 1.0. Exercises negative similarities end-to-end
        through evaluate(), not just the bare best_f1 metric.
        """
        import math

        class _AngleAdapter:
            """Encodes each text as a unit 2-D vector at a text-encoded angle.

            text_a -> 0 deg. text_b carries the intended cosine: 'pos' -> the
            angle whose cosine is -0.2, 'neg' -> the angle whose cosine is -0.8.
            """

            _ANGLES = {
                "anchor": 0.0,
                "pos": math.degrees(math.acos(-0.2)),
                "neg": math.degrees(math.acos(-0.8)),
            }

            def encode(self, texts, batch_size=64, normalize=True):
                out = np.empty((len(texts), 2), dtype=np.float32)
                for i, t in enumerate(texts):
                    theta = math.radians(self._ANGLES[t])
                    out[i] = (math.cos(theta), math.sin(theta))
                return out

            @property
            def dimension(self):
                return 2

            @property
            def name(self):
                return "angle-adapter"

        data = {
            "task": "neg_cos",
            "version": "0.1.0",
            "metric": "best_f1",
            "pairs": [
                {"id": "p1", "text_a": "anchor", "text_b": "pos", "label": 1},
                {"id": "p2", "text_a": "anchor", "text_b": "pos", "label": 1},
                {"id": "p3", "text_a": "anchor", "text_b": "neg", "label": 0},
                {"id": "p4", "text_a": "anchor", "text_b": "neg", "label": 0},
            ],
        }
        data_file = tmp_path / "neg_cos.json"
        data_file.write_text(json.dumps(data), encoding="utf-8")

        task = PairClassificationTask("neg_cos")
        task._data_path = data_file
        task.load_data()
        result = task.evaluate(_AngleAdapter())

        assert result.main_score == pytest.approx(1.0)
        assert result.details["best_threshold"] < 0.0

    def test_constant_adapter_gives_exact_degenerate_f1(
        self, tmp_path, sample_dedup_data, constant_adapter
    ):
        """ConstantAdapter pins best_f1 to an exact, hand-checked value.

        Every embedding is identical, so cosine similarity is 1.0 for all 6
        pairs (3 positive, 3 negative). The only candidate threshold is 1.0,
        which predicts every pair positive: tp=3, fp=3, fn=0, tn=0. That gives
        precision 0.5, recall 1.0, and F1 = 2*0.5*1.0/1.5 = 0.6667. Average
        precision collapses to the positive base rate, 0.5. A model with zero
        discriminative power can do no better, and this exact value would shift
        if the cosine or threshold-sweep logic regressed.
        """
        data_file = tmp_path / "test_dedup.json"
        data_file.write_text(json.dumps(sample_dedup_data), encoding="utf-8")

        task = PairClassificationTask("test_dedup")
        task._data_path = data_file
        task.load_data()
        result = task.evaluate(constant_adapter)

        assert result.main_score == pytest.approx(0.6667, abs=1e-4)
        assert result.details["best_threshold"] == pytest.approx(1.0)
        assert result.details["tp"] == 3
        assert result.details["fp"] == 3
        assert result.details["fn"] == 0
        assert result.details["tn"] == 0
        assert result.details["average_precision"] == pytest.approx(0.5)


# =========================================================================
# Classification task
# =========================================================================


class TestClassificationTask:
    """ClassificationTask loading, validation, and evaluation."""

    def test_load_valid_data(self, tmp_path, sample_classification_data):
        data_file = tmp_path / "test_classify.json"
        data_file.write_text(json.dumps(sample_classification_data), encoding="utf-8")

        task = ClassificationTask("test_classify")
        task._data_path = data_file
        task.load_data()
        assert len(task._items) == 12
        assert len(task._label_names) == 3

    def test_missing_text_raises(self, tmp_path):
        data = {
            "task": "bad_classify",
            "version": "0.1.0",
            "items": [{"id": "i001", "label": "Indian"}],
            "label_names": ["Indian"],
        }
        data_file = tmp_path / "bad_classify.json"
        data_file.write_text(json.dumps(data), encoding="utf-8")

        task = ClassificationTask("bad_classify")
        task._data_path = data_file
        with pytest.raises(ValueError, match="missing 'text'"):
            task.load_data()

    def test_missing_label_raises(self, tmp_path):
        data = {
            "task": "bad_classify",
            "version": "0.1.0",
            "items": [{"id": "i001", "text": "butter chicken"}],
            "label_names": ["Indian"],
        }
        data_file = tmp_path / "bad_classify.json"
        data_file.write_text(json.dumps(data), encoding="utf-8")

        task = ClassificationTask("bad_classify")
        task._data_path = data_file
        with pytest.raises(ValueError, match="missing 'label'"):
            task.load_data()

    def test_unknown_label_not_in_label_names_raises(self, tmp_path):
        data = {
            "task": "bad_classify",
            "version": "0.1.0",
            "items": [
                {"id": "i001", "text": "sushi", "label": "Japanese"},
            ],
            "label_names": ["Indian", "Italian"],
        }
        data_file = tmp_path / "bad_classify.json"
        data_file.write_text(json.dumps(data), encoding="utf-8")

        task = ClassificationTask("bad_classify")
        task._data_path = data_file
        with pytest.raises(ValueError, match="not found in label_names"):
            task.load_data()

    def test_empty_items_raises(self, tmp_path):
        data = {
            "task": "empty",
            "version": "0.1.0",
            "items": [],
            "label_names": [],
        }
        data_file = tmp_path / "empty.json"
        data_file.write_text(json.dumps(data), encoding="utf-8")

        task = ClassificationTask("empty")
        task._data_path = data_file
        with pytest.raises(ValueError, match="no items found"):
            task.load_data()

    def test_evaluate_returns_task_result(
        self, tmp_path, sample_classification_data, dummy_adapter
    ):
        data_file = tmp_path / "test_classify.json"
        data_file.write_text(json.dumps(sample_classification_data), encoding="utf-8")

        task = ClassificationTask("test_classify")
        task._data_path = data_file
        task.load_data()
        result = task.evaluate(dummy_adapter)

        assert isinstance(result, TaskResult)
        assert result.task_name == "test_classify"
        assert result.metric_name == "macro_f1"
        assert 0.0 <= result.main_score <= 1.0
        assert result.n_examples == 12

    def test_evaluate_details_structure(
        self, tmp_path, sample_classification_data, dummy_adapter
    ):
        data_file = tmp_path / "test_classify.json"
        data_file.write_text(json.dumps(sample_classification_data), encoding="utf-8")

        task = ClassificationTask("test_classify")
        task._data_path = data_file
        task.load_data()
        result = task.evaluate(dummy_adapter)

        assert "mean_macro_f1" in result.details
        assert "std_macro_f1" in result.details
        assert "mean_accuracy" in result.details
        assert "n_seeds" in result.details
        assert "per_class" in result.details
        assert "per_seed_f1" in result.details
        assert "label_names" in result.details

    def test_evaluate_runs_multiple_seeds(
        self, tmp_path, sample_classification_data, dummy_adapter
    ):
        data_file = tmp_path / "test_classify.json"
        data_file.write_text(json.dumps(sample_classification_data), encoding="utf-8")

        task = ClassificationTask("test_classify")
        task._data_path = data_file
        task.load_data()
        result = task.evaluate(dummy_adapter)

        n_seeds = result.details["n_seeds"]
        assert n_seeds == 10
        assert len(result.details["per_seed_f1"]) == n_seeds

    def test_evaluate_without_load_data_raises(self, dummy_adapter):
        """Calling evaluate before load_data should raise RuntimeError.

        Mirrors the guard already covered for retrieval and pair tasks, closing
        the gap for the classification task type.
        """
        task = ClassificationTask("unloaded_task")
        with pytest.raises(RuntimeError, match="load_data"):
            task.evaluate(dummy_adapter)

    def test_load_data_without_label_names(self, tmp_path):
        """When label_names is absent, load_data should infer labels from items."""
        data = {
            "task": "infer_labels",
            "version": "0.1.0",
            "items": [
                {"id": "i1", "text": "butter chicken", "label": "Indian"},
                {"id": "i2", "text": "pad thai", "label": "Thai"},
                {"id": "i3", "text": "pizza", "label": "Italian"},
                {"id": "i4", "text": "biryani", "label": "Indian"},
                {"id": "i5", "text": "risotto", "label": "Italian"},
                {"id": "i6", "text": "green curry", "label": "Thai"},
            ],
        }
        data_file = tmp_path / "infer_labels.json"
        data_file.write_text(json.dumps(data), encoding="utf-8")

        task = ClassificationTask("infer_labels")
        task._data_path = data_file
        task.load_data()
        assert len(task._items) == 6


# =========================================================================
# Task registry
# =========================================================================


class TestTaskDataLoading:
    """Edge cases in task data loading: malformed JSON and missing keys."""

    def test_malformed_json_raises_decode_error(self, tmp_path):
        """Invalid JSON should raise json.JSONDecodeError during load_data."""
        bad_file = tmp_path / "corrupt_search.json"
        bad_file.write_text("{not valid json!!!}", encoding="utf-8")

        task = RetrievalTask("corrupt_search")
        task._data_path = bad_file
        with pytest.raises(json.JSONDecodeError):
            task.load_data()

    def test_missing_corpus_key_raises(self, tmp_path):
        """Data dict without a 'corpus' key should raise a descriptive error."""
        data = {
            "task": "no_corpus",
            "version": "0.1.0",
            "queries": [{"id": "q1", "query": "curry", "relevance": {}}],
        }
        data_file = tmp_path / "no_corpus.json"
        data_file.write_text(json.dumps(data), encoding="utf-8")

        task = RetrievalTask("no_corpus")
        task._data_path = data_file
        with pytest.raises(KeyError):
            task.load_data()

    def test_missing_pairs_key_raises(self, tmp_path):
        """PairClassificationTask data without 'pairs' key should raise."""
        data = {"task": "no_pairs", "version": "0.1.0"}
        data_file = tmp_path / "no_pairs.json"
        data_file.write_text(json.dumps(data), encoding="utf-8")

        task = PairClassificationTask("no_pairs")
        task._data_path = data_file
        with pytest.raises(KeyError):
            task.load_data()

    def test_missing_items_key_raises(self, tmp_path):
        """ClassificationTask data without 'items' key should raise."""
        data = {"task": "no_items", "version": "0.1.0", "label_names": ["Indian"]}
        data_file = tmp_path / "no_items.json"
        data_file.write_text(json.dumps(data), encoding="utf-8")

        task = ClassificationTask("no_items")
        task._data_path = data_file
        with pytest.raises(KeyError):
            task.load_data()


class TestTaskRegistry:
    """Task registry: lookup, listing, and error handling."""

    def test_list_tasks_returns_sorted_names(self):
        from foodeval.tasks import list_tasks

        names = list_tasks()
        assert isinstance(names, list)
        assert len(names) >= 12
        assert names == sorted(names)

    def test_list_tasks_contains_known_tasks(self):
        from foodeval.tasks import list_tasks

        names = list_tasks()
        for expected in [
            "food_search",
            "concept_search",
            "diet_search",
            "noisy_search",
            "indian_match",
            "global_match",
            "beverage_match",
            "bakery_match",
            "portion_size",
            "noisy_menu_match",
            "cross_lingual_match",
            "cuisine_classify",
        ]:
            assert expected in names

    def test_get_task_returns_correct_type(self):
        from foodeval.tasks import get_task

        assert isinstance(get_task("food_search"), RetrievalTask)
        assert isinstance(get_task("indian_match"), PairClassificationTask)
        assert isinstance(get_task("cuisine_classify"), ClassificationTask)

    def test_get_task_unknown_raises(self):
        from foodeval.tasks import get_task

        with pytest.raises(KeyError, match="Unknown task"):
            get_task("nonexistent_task_xyz")

    def test_get_task_error_lists_available(self):
        """Error message for unknown task should list available task names."""
        from foodeval.tasks import get_task

        with pytest.raises(KeyError, match="food_search"):
            get_task("nonexistent_task_xyz")

    def test_get_all_tasks_returns_all(self):
        from foodeval.tasks import get_all_tasks, list_tasks

        tasks = get_all_tasks()
        assert len(tasks) == len(list_tasks())
        for task in tasks:
            assert isinstance(task, BenchmarkTask)

    def test_task_names_match_data_files(self):
        """Every registered task should have a corresponding data file."""
        from foodeval.tasks import list_tasks

        for name in list_tasks():
            data_path = DATA_DIR / f"{name}.json"
            assert data_path.exists(), f"Missing data file for task: {name}"

    def test_all_retrieval_tasks_have_correct_type(self):
        from foodeval.tasks import get_task

        for name in ["food_search", "concept_search", "diet_search", "noisy_search"]:
            assert get_task(name).task_type == "retrieval"

    def test_all_pair_tasks_have_correct_type(self):
        from foodeval.tasks import get_task

        for name in [
            "indian_match",
            "global_match",
            "beverage_match",
            "bakery_match",
            "portion_size",
            "noisy_menu_match",
            "cross_lingual_match",
        ]:
            assert get_task(name).task_type == "pair_classification"

    def test_classification_tasks_have_correct_type(self):
        from foodeval.tasks import get_task

        assert get_task("cuisine_classify").task_type == "classification"
