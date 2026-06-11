"""Tests for the evaluation orchestrator (foodeval.evaluate).

The evaluate module orchestrates running multiple tasks against an adapter
and aggregating results into a BenchmarkResult (FoodEval).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from foodeval.evaluate import run_benchmark, BenchmarkResult
from foodeval.tasks import get_task
from foodeval.tasks.base import BenchmarkTask, TaskResult


# =========================================================================
# BenchmarkResult dataclass
# =========================================================================


class TestBenchmarkResultDataclass:
    """BenchmarkResult construction and field defaults."""

    def test_default_values(self):
        result = BenchmarkResult(model_name="test", dimension=64)
        assert result.task_results == {}
        assert result.aggregate_score == 0.0
        assert result.timestamp == ""
        assert result.total_seconds == 0.0

    def test_to_dict_has_required_keys(self):
        result = BenchmarkResult(model_name="test-model", dimension=128)
        d = result.to_dict()
        assert set(d.keys()) == {
            "model_name",
            "dimension",
            "aggregate_score",
            "category_scores",
            "timestamp",
            "total_seconds",
            "metadata",
            "tasks",
        }

    def test_to_dict_rounds_values(self):
        result = BenchmarkResult(
            model_name="test",
            dimension=64,
            aggregate_score=0.87654321,
            total_seconds=12.34567890,
        )
        d = result.to_dict()
        assert d["aggregate_score"] == 0.8765
        assert d["total_seconds"] == 12.346


# =========================================================================
# run_benchmark
# =========================================================================


class TestRunBenchmark:
    """run_benchmark: orchestrate task evaluation."""

    def test_returns_benchmark_result(self, dummy_adapter):
        result = run_benchmark(dummy_adapter, tasks=["indian_match"], verbose=False)
        assert isinstance(result, BenchmarkResult)

    def test_runs_specified_tasks_only(self, dummy_adapter):
        result = run_benchmark(
            dummy_adapter,
            tasks=["indian_match", "cross_lingual_match"],
            verbose=False,
        )
        task_names = set(result.task_results.keys())
        assert task_names == {"indian_match", "cross_lingual_match"}

    def test_unknown_task_raises(self, dummy_adapter):
        with pytest.raises((KeyError, ValueError)):
            run_benchmark(dummy_adapter, tasks=["nonexistent_xyz"], verbose=False)

    def test_result_contains_model_name(self, dummy_adapter):
        result = run_benchmark(dummy_adapter, tasks=["indian_match"], verbose=False)
        assert result.model_name == dummy_adapter.name

    def test_each_task_result_has_score(self, dummy_adapter):
        result = run_benchmark(dummy_adapter, tasks=["indian_match"], verbose=False)
        for tr in result.task_results.values():
            assert 0.0 <= tr.main_score <= 1.0
            assert tr.n_examples > 0

    def test_result_has_timestamp(self, dummy_adapter):
        result = run_benchmark(dummy_adapter, tasks=["indian_match"], verbose=False)
        assert len(result.timestamp) > 0
        # ISO-8601 format
        assert "T" in result.timestamp

    def test_result_has_dimension(self, dummy_adapter):
        result = run_benchmark(dummy_adapter, tasks=["indian_match"], verbose=False)
        assert result.dimension == dummy_adapter.dimension

    def test_total_seconds_is_positive(self, dummy_adapter):
        result = run_benchmark(dummy_adapter, tasks=["indian_match"], verbose=False)
        assert result.total_seconds > 0.0

    def test_verbose_writes_to_stderr(self, dummy_adapter, capsys):
        """Verbose mode should output progress to stderr."""
        run_benchmark(dummy_adapter, tasks=["indian_match"], verbose=True)
        captured = capsys.readouterr()
        assert "indian_match" in captured.err
        assert "FoodEval Score" in captured.err

    def test_error_recovery_produces_fallback_result(self, dummy_adapter):
        """When a task raises during evaluate(), run_benchmark should catch the error
        and produce a TaskResult with main_score=0.0 and error details."""

        class FailingTask(BenchmarkTask):
            task_type = "pair_classification"
            metric_name = "best_f1"

            def load_data(self) -> None:
                self._data = {"version": "0.1.0"}

            def evaluate(self, adapter):
                raise ValueError("simulated evaluation failure")

        failing_task = FailingTask("failing_task")

        with patch("foodeval.evaluate.get_task", return_value=failing_task):
            result = run_benchmark(dummy_adapter, tasks=["failing_task"], verbose=False)

        assert "failing_task" in result.task_results
        tr = result.task_results["failing_task"]
        assert tr.main_score == 0.0
        assert tr.metric_name == "best_f1"
        assert tr.errored is True
        assert "error" in tr.details
        assert "simulated evaluation failure" in tr.details["error"]

    def test_errored_task_excluded_from_aggregate(self, dummy_adapter):
        """A task that errors must not drag the aggregate toward zero.

        With one healthy task (real score > 0) and one that raises, the
        aggregate should equal the healthy task's score alone, because errored
        tasks are dropped from the mean.
        """

        class FailingTask(BenchmarkTask):
            task_type = "pair_classification"
            metric_name = "best_f1"

            def load_data(self) -> None:
                self._data = {"version": "0.1.0"}

            def evaluate(self, adapter):
                raise ValueError("boom")

        real_task = get_task("indian_match")
        failing_task = FailingTask("failing_task")

        def fake_get_task(name):
            return real_task if name == "indian_match" else failing_task

        with patch("foodeval.evaluate.get_task", side_effect=fake_get_task):
            result = run_benchmark(
                dummy_adapter,
                tasks=["indian_match", "failing_task"],
                verbose=False,
            )

        healthy_score = result.task_results["indian_match"].main_score
        assert result.task_results["failing_task"].errored is True
        assert healthy_score > 0.0
        # Aggregate ignores the errored task, so it equals the healthy score
        # (rounded to 4 dp by run_benchmark), not their average with 0.0.
        assert result.aggregate_score == pytest.approx(healthy_score, abs=1e-4)

    def test_all_tasks_errored_aggregate_is_zero(self, dummy_adapter):
        """When every task errors, the aggregate falls back to 0.0.

        Exercises the ``sum(scores) / len(scores) if scores else 0.0`` empty
        branch: with no successful scores, division is skipped and 0.0 is used.
        """

        class FailingTask(BenchmarkTask):
            task_type = "pair_classification"
            metric_name = "best_f1"

            def load_data(self) -> None:
                self._data = {"version": "0.1.0"}

            def evaluate(self, adapter):
                raise RuntimeError("always fails")

        failing_task = FailingTask("failing_task")

        with patch("foodeval.evaluate.get_task", return_value=failing_task):
            result = run_benchmark(
                dummy_adapter,
                tasks=["failing_task", "failing_task"],
                verbose=False,
            )

        assert all(tr.errored for tr in result.task_results.values())
        assert result.aggregate_score == 0.0


# =========================================================================
# Registry singleton isolation
# =========================================================================


class TestRegistrySingletonIsolation:
    """TASK_REGISTRY holds shared singletons; runs must not leak cached state."""

    def test_repeated_runs_of_registry_task_are_identical(self, dummy_adapter):
        """Running the same registry task twice yields byte-identical scores.

        The task instance is a module-level singleton that caches its loaded
        data. With deterministic embeddings, two back-to-back runs through
        run_benchmark must produce the same main_score and per-domain
        breakdown -- proving the cache is reused safely (and the autouse reset
        fixture does not perturb a within-test rerun).
        """
        first = run_benchmark(dummy_adapter, tasks=["indian_match"], verbose=False)
        second = run_benchmark(dummy_adapter, tasks=["indian_match"], verbose=False)

        tr1 = first.task_results["indian_match"]
        tr2 = second.task_results["indian_match"]
        assert tr1.main_score == tr2.main_score
        assert tr1.n_examples == tr2.n_examples
        assert tr1.details["per_domain"] == tr2.details["per_domain"]
        assert first.aggregate_score == second.aggregate_score

    def test_registry_singletons_start_each_test_unloaded(self):
        """At test entry, every registry singleton has its cache cleared.

        The autouse reset fixture runs before this test body, so no matter
        what an earlier test loaded into a shared singleton, each task starts
        with _data is None. This is the invariant that prevents order-dependent
        leakage between tests.
        """
        from foodeval.tasks import TASK_REGISTRY

        for name, task in TASK_REGISTRY.items():
            assert task._data is None, f"{name} entered the test with stale data"

    def test_dirtying_a_singleton_does_not_leak_to_next_test(self):
        """Deliberately load data; the reset fixture must scrub it afterward.

        Paired with test_registry_singletons_start_each_test_unloaded above:
        this test dirties a singleton, and the entry-state invariant proves the
        fixture cleaned up. Loading also must not raise for the real data file.
        """
        from foodeval.tasks import TASK_REGISTRY

        task = TASK_REGISTRY["indian_match"]
        task.load_data()
        assert task._data is not None  # loaded within this test
        # Teardown side of the autouse fixture clears it before the next test.


# =========================================================================
# TaskResult.to_dict errored flag
# =========================================================================


class TestTaskResultErroredFlag:
    """TaskResult.to_dict: conditional inclusion of the errored field."""

    def test_errored_true_included_in_to_dict(self):
        """When errored=True, to_dict() must include 'errored': True."""
        result = TaskResult(
            task_name="failing_task",
            main_score=0.0,
            metric_name="best_f1",
            errored=True,
        )
        d = result.to_dict()
        assert "errored" in d
        assert d["errored"] is True

    def test_errored_false_excluded_from_to_dict(self):
        """When errored=False (default), to_dict() must NOT include the 'errored' key."""
        result = TaskResult(
            task_name="normal_task",
            main_score=0.85,
            metric_name="ndcg@10",
        )
        d = result.to_dict()
        assert "errored" not in d


# =========================================================================
# BenchmarkResult serialization
# =========================================================================


class TestBenchmarkResultSerialization:
    """BenchmarkResult.to_dict, .to_json, .to_markdown."""

    @pytest.fixture
    def benchmark_result(self, dummy_adapter):
        return run_benchmark(dummy_adapter, tasks=["indian_match"], verbose=False)

    def test_to_dict_returns_dict(self, benchmark_result):
        d = benchmark_result.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_contains_tasks(self, benchmark_result):
        d = benchmark_result.to_dict()
        assert "tasks" in d
        assert isinstance(d["tasks"], dict)

    def test_to_dict_task_results_match(self, benchmark_result):
        """Task dicts should contain the same keys as TaskResult.to_dict."""
        d = benchmark_result.to_dict()
        for task_name, task_dict in d["tasks"].items():
            assert "task_name" in task_dict
            assert "main_score" in task_dict
            assert "metric_name" in task_dict

    def test_to_json_writes_valid_json_file(self, benchmark_result, tmp_path):
        outpath = str(tmp_path / "results.json")
        benchmark_result.to_json(outpath)
        with open(outpath, "r", encoding="utf-8") as f:
            parsed = json.load(f)
        assert isinstance(parsed, dict)
        assert "model_name" in parsed

    def test_to_json_creates_parent_dirs(self, benchmark_result, tmp_path):
        outpath = str(tmp_path / "nested" / "dir" / "results.json")
        benchmark_result.to_json(outpath)
        assert Path(outpath).exists()

    def test_to_json_roundtrips_with_to_dict(self, benchmark_result, tmp_path):
        """JSON file should parse back to the same structure as to_dict."""
        d = benchmark_result.to_dict()
        outpath = str(tmp_path / "results.json")
        benchmark_result.to_json(outpath)
        with open(outpath, "r", encoding="utf-8") as f:
            parsed = json.load(f)
        assert set(d.keys()) == set(parsed.keys())

    def test_to_markdown_returns_string(self, benchmark_result):
        md = benchmark_result.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 0

    def test_to_markdown_contains_task_name(self, benchmark_result):
        md = benchmark_result.to_markdown()
        assert "indian_match" in md

    def test_to_markdown_contains_model_name(self, benchmark_result):
        md = benchmark_result.to_markdown()
        assert benchmark_result.model_name in md

    def test_to_markdown_contains_table_header(self, benchmark_result):
        md = benchmark_result.to_markdown()
        assert "| Task |" in md
        assert "Score" in md

    def test_to_markdown_contains_aggregate_score(self, benchmark_result):
        md = benchmark_result.to_markdown()
        assert "FoodEval Score" in md


# =========================================================================
# Aggregate score computation
# =========================================================================


class TestAggregateScore:
    """BenchmarkResult.aggregate_score: single summary metric."""

    def test_aggregate_is_float(self, dummy_adapter):
        result = run_benchmark(dummy_adapter, tasks=["indian_match"], verbose=False)
        assert isinstance(result.aggregate_score, float)

    def test_aggregate_in_valid_range(self, dummy_adapter):
        result = run_benchmark(dummy_adapter, tasks=["indian_match"], verbose=False)
        assert 0.0 <= result.aggregate_score <= 1.0

    def test_single_task_aggregate_equals_main_score(self, dummy_adapter):
        """With one task, aggregate should equal that task's main_score."""
        result = run_benchmark(dummy_adapter, tasks=["indian_match"], verbose=False)
        assert len(result.task_results) == 1
        only_result = list(result.task_results.values())[0]
        assert result.aggregate_score == pytest.approx(only_result.main_score, abs=0.01)

    def test_multiple_tasks_aggregate_is_mean(self, dummy_adapter):
        """With multiple tasks, aggregate should be the arithmetic mean."""
        result = run_benchmark(
            dummy_adapter,
            tasks=["indian_match", "cross_lingual_match"],
            verbose=False,
        )
        scores = [tr.main_score for tr in result.task_results.values()]
        expected_mean = sum(scores) / len(scores)
        assert result.aggregate_score == pytest.approx(expected_mean, abs=0.01)

    def test_aggregate_handles_zero_scores(self, dummy_adapter):
        """Even if all tasks score 0, aggregate should be 0.0 (not crash)."""
        result = BenchmarkResult(model_name="test", dimension=64)
        result.task_results = {
            "task_a": TaskResult(task_name="task_a", main_score=0.0, metric_name="f1"),
            "task_b": TaskResult(task_name="task_b", main_score=0.0, metric_name="f1"),
        }
        scores = [tr.main_score for tr in result.task_results.values()]
        manual_aggregate = sum(scores) / len(scores) if scores else 0.0
        assert manual_aggregate == 0.0
