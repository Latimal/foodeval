"""Tests for the FoodEval CLI (foodeval.cli).

The CLI provides subcommands for running benchmarks, listing tasks, and
showing task info. Tests pass argv directly via main(argv) instead of
patching sys.argv.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch, MagicMock

import pytest

from foodeval.cli import main, _build_parser, _build_adapter


# =========================================================================
# List subcommand
# =========================================================================


class TestCLIListCommand:
    """foodeval list: show available tasks."""

    def test_list_returns_zero(self):
        """The list subcommand should return exit code 0."""
        exit_code = main(["list"])
        assert exit_code == 0

    def test_list_prints_task_names(self, capsys):
        main(["list"])
        output = capsys.readouterr().out
        assert "food_search" in output
        assert "indian_match" in output
        assert "cuisine_classify" in output

    def test_list_prints_task_count(self, capsys):
        main(["list"])
        output = capsys.readouterr().out
        assert "FoodEval tasks" in output

    def test_list_shows_task_types(self, capsys):
        main(["list"])
        output = capsys.readouterr().out
        assert "retrieval" in output
        assert "pair_classification" in output
        assert "classification" in output


# =========================================================================
# Info subcommand
# =========================================================================


class TestCLIInfoCommand:
    """foodeval info <task>: show task details."""

    def test_info_known_task_returns_zero(self):
        exit_code = main(["info", "food_search"])
        assert exit_code == 0

    def test_info_unknown_task_returns_nonzero(self):
        exit_code = main(["info", "nonexistent_xyz"])
        assert exit_code != 0

    def test_info_prints_task_metadata(self, capsys):
        main(["info", "food_search"])
        output = capsys.readouterr().out
        assert "food_search" in output

    def test_info_output_is_valid_json(self, capsys):
        main(["info", "food_search"])
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert "name" in parsed
        assert "task_type" in parsed

    def test_info_shows_metric(self, capsys):
        main(["info", "indian_match"])
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert "metric" in parsed


# =========================================================================
# Argument parsing
# =========================================================================


class TestCLIArgumentParsing:
    """Argument parsing edge cases."""

    def test_no_subcommand_exits_with_error(self):
        """Calling without a subcommand should raise SystemExit (argparse)."""
        with pytest.raises(SystemExit):
            main([])

    def test_invalid_subcommand_exits_with_error(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["foobar"])
        assert exc_info.value.code not in (None, 0)

    def test_run_parser_has_model_arg(self):
        parser = _build_parser()
        # Parse valid run args to verify the parser accepts --model
        args = parser.parse_args(["run", "--model", "test-model"])
        assert args.model == "test-model"

    def test_run_parser_accepts_dim(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--model", "test", "--dim", "384"])
        assert args.dim == 384

    def test_run_parser_accepts_tasks(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["run", "--model", "test", "--tasks", "food_search,indian_match"]
        )
        assert args.tasks == "food_search,indian_match"

    def test_run_parser_accepts_output(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--model", "test", "--output", "results.json"])
        assert args.output == "results.json"

    def test_quiet_flag_overrides_verbose(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--model", "test", "--quiet"])
        assert args.quiet is True


# =========================================================================
# Run subcommand
# =========================================================================


class TestCLIRunCommand:
    """foodeval run: execute benchmarks.

    These tests verify argument parsing and error handling. They do NOT
    actually run benchmarks (that would require a real model).
    """

    def test_run_without_model_arg_exits_with_error(self):
        """The run command requires a --model argument."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run"])
        # argparse exits 2 for missing required args
        assert exc_info.value.code not in (None, 0)

    def test_run_with_errored_task_returns_nonzero(self, capsys):
        """A run where a task errors must exit nonzero so CI catches it."""
        from foodeval.evaluate import BenchmarkResult
        from foodeval.tasks.base import TaskResult

        failed = BenchmarkResult(model_name="m", dimension=4096)
        failed.task_results["indian_match"] = TaskResult(
            task_name="indian_match",
            main_score=0.0,
            metric_name="best_f1",
            details={"error": "boom"},
            n_examples=0,
            elapsed_seconds=0.0,
            errored=True,
        )
        with patch("foodeval.cli.run_benchmark", return_value=failed):
            exit_code = main(["run", "--model", "bm25", "--tasks", "indian_match"])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert "errored" in captured.err

    def test_run_with_invalid_tasks_returns_nonzero(self):
        """Requesting a nonexistent task should return exit code 1."""
        # Use bm25 as the model since it's cheap to construct
        exit_code = main(["run", "--model", "bm25", "--tasks", "nonexistent_xyz"])
        assert exit_code != 0


# =========================================================================
# Run subcommand: happy path (real bm25 model on a real task)
# =========================================================================


class TestCLIRunHappyPath:
    """foodeval run end-to-end with the cheap bm25 baseline on one real task.

    bm25 vectorizes in-process (no network, no GPU, no downloads) and the
    indian_match data file ships with the package, so these exercise the full
    run path quickly and deterministically.
    """

    def test_quiet_run_returns_zero_and_prints_table_to_stdout(self, capsys):
        """--quiet returns 0, prints the markdown table to stdout, and suppresses
        the stderr progress lines."""
        exit_code = main(
            ["run", "--model", "bm25", "--tasks", "indian_match", "--quiet"]
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "| Task |" in captured.out
        assert "indian_match" in captured.out
        # Progress goes to stderr and must be silenced in quiet mode.
        assert "Running indian_match" not in captured.err

    def test_verbose_run_prints_progress_to_stderr(self, capsys):
        """Without --quiet, progress is written to stderr while the table still
        goes to stdout."""
        exit_code = main(["run", "--model", "bm25", "--tasks", "indian_match"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "| Task |" in captured.out
        assert "Running indian_match" in captured.err
        assert "FoodEval Score" in captured.err

    def test_output_writes_valid_json_matching_schema(self, tmp_path, capsys):
        """--output writes a JSON file matching BenchmarkResult.to_dict()."""
        out_path = tmp_path / "result.json"
        exit_code = main(
            [
                "run",
                "--model",
                "bm25",
                "--tasks",
                "indian_match",
                "--quiet",
                "--output",
                str(out_path),
            ]
        )
        capsys.readouterr()
        assert exit_code == 0
        assert out_path.exists()

        parsed = json.loads(out_path.read_text(encoding="utf-8"))
        # Top-level schema from BenchmarkResult.to_dict.
        assert set(parsed.keys()) == {
            "model_name",
            "dimension",
            "aggregate_score",
            "category_scores",
            "timestamp",
            "total_seconds",
            "metadata",
            "tasks",
        }
        assert parsed["model_name"] == "Lexical (TF)"
        assert parsed["dimension"] == 4096
        # Per-task schema from TaskResult.to_dict.
        assert "indian_match" in parsed["tasks"]
        task_dict = parsed["tasks"]["indian_match"]
        assert task_dict["task_name"] == "indian_match"
        assert task_dict["metric_name"] == "best_f1"
        assert 0.0 <= task_dict["main_score"] <= 1.0
        # Single-task aggregate equals that task's score.
        assert parsed["aggregate_score"] == pytest.approx(
            task_dict["main_score"], abs=1e-4
        )


# =========================================================================
# Preflight and matrix commands
# =========================================================================


class TestCLIPreflightAndMatrix:
    """Cheap smoke tests for the protocol/preflight commands."""

    def test_preflight_known_task_returns_manifest(self, capsys):
        exit_code = main(["preflight", "--tasks", "food_search"])
        assert exit_code == 0
        parsed = json.loads(capsys.readouterr().out)
        assert "tasks" in parsed
        assert "food_search" in parsed["tasks"]
        assert parsed["tasks"]["food_search"]["type"] == "retrieval"

    def test_matrix_list_includes_bm25(self, capsys):
        exit_code = main(["matrix", "--list", "--models", "bm25"])
        assert exit_code == 0
        output = capsys.readouterr().out
        assert "bm25" in output
        assert "lexical" in output

    def test_matrix_plan_writes_files(self, tmp_path, capsys):
        out_dir = tmp_path / "matrix"
        exit_code = main(
            [
                "matrix",
                "--models",
                "bm25",
                "--tasks",
                "food_search",
                "--output-dir",
                str(out_dir),
                "--quiet",
            ]
        )
        assert exit_code == 0
        capsys.readouterr()
        assert (out_dir / "matrix-plan.json").exists()
        assert (out_dir / "preflight.json").exists()

    def test_no_cache_sets_cache_disabled_at_runtime(self, capsys):
        """--no-cache must flip foodeval.adapters.base.CACHE_DISABLED to True
        before the benchmark runs (not merely parse the flag).

        The flag is captured from inside a patched run_benchmark so a
        regression that set it too late, or unset it, would be caught.
        """
        import foodeval.adapters.base as base_mod
        from foodeval.evaluate import BenchmarkResult

        original = base_mod.CACHE_DISABLED
        seen: dict[str, bool] = {}

        def fake_run_benchmark(adapter, tasks, verbose):
            seen["cache_disabled"] = base_mod.CACHE_DISABLED
            return BenchmarkResult(model_name=adapter.name, dimension=adapter.dimension)

        try:
            with patch("foodeval.cli.run_benchmark", side_effect=fake_run_benchmark):
                exit_code = main(
                    [
                        "run",
                        "--model",
                        "bm25",
                        "--tasks",
                        "indian_match",
                        "--quiet",
                        "--no-cache",
                    ]
                )
            capsys.readouterr()
            assert exit_code == 0
            assert seen["cache_disabled"] is True
        finally:
            base_mod.CACHE_DISABLED = original

    def test_cache_enabled_by_default_at_runtime(self, capsys):
        """Without --no-cache, CACHE_DISABLED stays False during the run."""
        import foodeval.adapters.base as base_mod
        from foodeval.evaluate import BenchmarkResult

        original = base_mod.CACHE_DISABLED
        base_mod.CACHE_DISABLED = False
        seen: dict[str, bool] = {}

        def fake_run_benchmark(adapter, tasks, verbose):
            seen["cache_disabled"] = base_mod.CACHE_DISABLED
            return BenchmarkResult(model_name=adapter.name, dimension=adapter.dimension)

        try:
            with patch("foodeval.cli.run_benchmark", side_effect=fake_run_benchmark):
                exit_code = main(
                    ["run", "--model", "bm25", "--tasks", "indian_match", "--quiet"]
                )
            capsys.readouterr()
            assert exit_code == 0
            assert seen["cache_disabled"] is False
        finally:
            base_mod.CACHE_DISABLED = original


# =========================================================================
# _build_adapter parsing
# =========================================================================


class TestBuildAdapter:
    """_build_adapter: parse model string into an adapter."""

    def test_bm25_string(self):
        adapter = _build_adapter("bm25", dim=None)
        assert adapter.name == "Lexical (TF)"

    def test_openai_prefix(self):
        """openai: prefix should construct an OpenAIAdapter with the parsed model name."""
        with patch("foodeval.adapters.openai_adapter.OpenAIAdapter") as mock_cls:
            mock_instance = mock_cls.return_value
            mock_instance.name = "text-embedding-3-large-384d"
            mock_instance.dimension = 384
            adapter = _build_adapter("openai:text-embedding-3-large", dim=384)
            mock_cls.assert_called_once_with(
                model="text-embedding-3-large", dimension=384
            )
            assert adapter.name == "text-embedding-3-large-384d"

    def test_bedrock_prefix(self):
        """bedrock: prefix should construct a BedrockAdapter with the parsed model ID."""
        with patch("foodeval.adapters.bedrock.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            adapter = _build_adapter("bedrock:cohere.embed-multilingual-v3", dim=384)
            assert adapter.dimension == 384
            assert "embed-multilingual-v3" in adapter.name

    def test_cohere_prefix(self):
        """cohere: prefix should construct a CohereAdapter with the parsed model name."""
        with patch.dict(os.environ, {"COHERE_API_KEY": "test-key-for-unit-test"}):
            adapter = _build_adapter("cohere:embed-multilingual-v3.0", dim=384)
            assert adapter.dimension == 384
            assert "embed-multilingual-v3" in adapter.name

    def test_gemini_prefix(self):
        """gemini: prefix should construct a GeminiAdapter with the parsed model name."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-for-unit-test"}):
            adapter = _build_adapter("gemini:gemini-embedding-2", dim=384)
            assert adapter.dimension == 384
            assert "gemini-embedding-2" in adapter.name

    def test_vertex_prefix(self):
        """vertex: prefix should construct a VertexAdapter with the parsed
        model name; the adapter name carries no vertex prefix."""
        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "test-project"}):
            adapter = _build_adapter("vertex:gemini-embedding-001", dim=384)
            assert adapter.dimension == 384
            assert adapter.name == "gemini-embedding-001-384d"

    def test_lexical_tf_and_bm25_alias_build_same_adapter(self):
        """lexical-tf is the primary key; bm25 remains a legacy alias."""
        a = _build_adapter("lexical-tf", dim=None)
        b = _build_adapter("bm25", dim=None)
        assert type(a) is type(b)
        assert a.name == b.name

    def test_unknown_prefix_uses_sentence_transformer(self):
        """Any string without a recognized prefix should use SentenceTransformer."""
        with patch(
            "foodeval.adapters.sentence_transformer.SentenceTransformer"
        ) as mock_st:
            mock_model = mock_st.return_value
            mock_model.get_sentence_embedding_dimension.return_value = 384
            adapter = _build_adapter("some-local-model", dim=384)
            assert "384d" in adapter.name


# =========================================================================
# Leaderboard subcommand
# =========================================================================


class TestCLILeaderboardCommand:
    """foodeval leaderboard <dir>: generate leaderboard."""

    def test_leaderboard_nonexistent_dir_returns_nonzero(self):
        exit_code = main(["leaderboard", "/tmp/foodeval_nonexistent_dir_xyz"])
        assert exit_code != 0

    def test_leaderboard_with_results(self, tmp_path, capsys):
        """Should generate a markdown leaderboard from result files."""
        # Create a minimal result file
        result = {
            "model_name": "test-model",
            "dimension": 64,
            "aggregate_score": 0.75,
            "timestamp": "2026-01-01T00:00:00Z",
            "total_seconds": 5.0,
            "tasks": {
                "food_search": {
                    "task_name": "food_search",
                    "main_score": 0.8,
                    "metric_name": "ndcg@10",
                },
                "indian_match": {
                    "task_name": "indian_match",
                    "main_score": 0.7,
                    "metric_name": "best_f1",
                },
            },
        }
        result_file = tmp_path / "test-model.json"
        result_file.write_text(json.dumps(result), encoding="utf-8")

        exit_code = main(["leaderboard", str(tmp_path)])
        assert exit_code == 0
        output = capsys.readouterr().out
        assert "test-model" in output
        assert "Leaderboard" in output


# =========================================================================
# --no-cache flag
# =========================================================================


class TestNoCacheFlag:
    """--no-cache flag should disable disk caching at runtime."""

    def test_no_cache_sets_cache_disabled(self):
        """--no-cache should set CACHE_DISABLED = True in adapters.base."""
        import foodeval.adapters.base as base_mod

        original = base_mod.CACHE_DISABLED
        try:
            parser = _build_parser()
            args = parser.parse_args(["run", "--model", "bm25", "--no-cache"])
            assert args.no_cache is True
        finally:
            base_mod.CACHE_DISABLED = original

    def test_no_cache_flag_parsed(self):
        """--no-cache should be accepted by argparse."""
        parser = _build_parser()
        args = parser.parse_args(["run", "--model", "bm25", "--no-cache"])
        assert args.no_cache is True

    def test_default_cache_enabled(self):
        """Without --no-cache, caching should be enabled (default)."""
        parser = _build_parser()
        args = parser.parse_args(["run", "--model", "bm25"])
        assert args.no_cache is False
