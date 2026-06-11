"""FoodEval command-line interface.

Provides the ``foodeval`` CLI with subcommands for running evaluations,
listing tasks, inspecting task details, and generating leaderboards.

Usage:
    # Run all tasks with a local model
    python -m foodeval run --model BAAI/bge-m3 --dim 384

    # Run specific tasks with an API model
    python -m foodeval run --model openai:text-embedding-3-large --tasks food_search,indian_match

    # Run with BM25 baseline
    python -m foodeval run --model bm25

    # List available tasks
    python -m foodeval list

    # Show task details
    python -m foodeval info food_search

    # Generate leaderboard from saved results
    python -m foodeval leaderboard results/
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

from foodeval import __version__
from foodeval.adapters.base import EmbeddingAdapter
from foodeval.adapters.factory import build_adapter as _factory_build_adapter
from foodeval.evaluate import run_benchmark
from foodeval.leaderboard import generate_leaderboard
from foodeval.matrix import load_matrix, matrix_runs, run_matrix, write_default_matrix
from foodeval.preflight import contamination_report, write_report
from foodeval.tasks import get_task, list_tasks


def _build_adapter(model_str: str, dim: int | None) -> EmbeddingAdapter:
    """Backward-compatible wrapper around the shared adapter factory."""
    return _factory_build_adapter(model_str, dim)


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute the 'run' subcommand."""
    verbose = not args.quiet

    if args.no_cache:
        import foodeval.adapters.base as _base

        _base.CACHE_DISABLED = True

    try:
        adapter = _build_adapter(args.model, args.dim)
    except (ImportError, ValueError) as exc:
        print(f"Error loading model: {exc}", file=sys.stderr)
        if verbose:
            traceback.print_exc()
        return 1

    task_names: list[str] | None = None
    if args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",")]
        # Validate
        valid = set(list_tasks())
        invalid = [t for t in task_names if t not in valid]
        if invalid:
            print(
                f"Unknown tasks: {', '.join(invalid)}. "
                f"Available: {', '.join(sorted(valid))}",
                file=sys.stderr,
            )
            return 1

    try:
        result = run_benchmark(
            adapter=adapter,
            tasks=task_names,
            verbose=verbose,
        )
    except FileNotFoundError as exc:
        print(f"Data error: {exc}", file=sys.stderr)
        if verbose:
            traceback.print_exc()
        return 1
    except Exception as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        if verbose:
            traceback.print_exc()
        return 1

    # Output
    if args.output:
        result.to_json(args.output)
        if verbose:
            print(f"\nResults saved to {args.output}", file=sys.stderr)

    # Always print markdown summary to stdout
    print(result.to_markdown())

    errored = [name for name, tr in result.task_results.items() if tr.errored]
    if errored:
        print(
            f"{len(errored)} task(s) errored: {', '.join(errored)}",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    """Execute the 'list' subcommand."""
    tasks = list_tasks()
    print(f"FoodEval tasks ({len(tasks)}):\n")
    for name in tasks:
        task = get_task(name)
        print(f"  {name:25s}  {task.task_type:20s}  {task.metric_name}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    """Execute the 'info' subcommand."""
    try:
        task = get_task(args.task_name)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        info = task.describe()
    except FileNotFoundError as exc:
        print(f"Cannot load task data: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


def _cmd_leaderboard(args: argparse.Namespace) -> int:
    """Execute the 'leaderboard' subcommand."""
    try:
        markdown = generate_leaderboard(args.results_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(markdown)
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    """Execute the 'preflight' subcommand."""
    task_names = None
    if args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    report = contamination_report(args.compare or [], task_names=task_names)
    if args.output:
        write_report(report, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _cmd_matrix(args: argparse.Namespace) -> int:
    """Execute the 'matrix' subcommand."""
    task_names = None
    if args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    model_ids = None
    if args.models:
        model_ids = {m.strip() for m in args.models.split(",") if m.strip()}
    dims = None
    if args.dims:
        dims = {int(d.strip()) for d in args.dims.split(",") if d.strip()}

    if args.write_default_config:
        write_default_matrix(args.write_default_config)
        print(f"Wrote default matrix to {args.write_default_config}")
        return 0

    matrix = load_matrix(args.config)
    if args.list:
        runs = matrix_runs(matrix, model_ids=model_ids, dims=dims)
        for run in runs:
            dim = "native" if run.dimension is None else f"{run.dimension}d"
            print(
                f"{run.id:42s} {run.entry.get('tier', 'unknown'):16s} "
                f"{run.entry['model']} @ {dim}"
            )
        return 0

    summary = run_matrix(
        output_dir=args.output_dir,
        tasks=task_names,
        config_path=args.config,
        model_ids=model_ids,
        dims=dims,
        compare_paths=args.compare or [],
        env_files=args.env_file or [],
        execute=args.execute,
        skip_missing_env=not args.fail_missing_env,
        verbose=not args.quiet,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.execute:
        print(
            "\nPlan only. Re-run with --execute to evaluate models.",
            file=sys.stderr,
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="foodeval",
        description="FoodEval: benchmark for food domain text embeddings.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"foodeval {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_parser = subparsers.add_parser(
        "run", help="Run benchmark evaluation on a model."
    )
    run_parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model to evaluate. Formats: 'BAAI/bge-m3' (local/HF), "
            "'openai:text-embedding-3-large', 'cohere:embed-v4.0', "
            "'voyage:voyage-4-large', 'gemini:gemini-embedding-2', "
            "'vertex:gemini-embedding-001', "
            "'bedrock:cohere.embed-multilingual-v3', 'lexical-tf' (alias 'bm25')."
        ),
    )
    run_parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Embedding dimension (Matryoshka truncation for local models, "
        "dimension override for API models).",
    )
    run_parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of tasks to run (default: all).",
    )
    run_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON.",
    )
    run_parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress progress output.",
    )
    run_parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable disk cache for embeddings.",
    )
    run_parser.set_defaults(func=_cmd_run)

    # --- list ---
    list_parser = subparsers.add_parser("list", help="List available benchmark tasks.")
    list_parser.set_defaults(func=_cmd_list)

    # --- info ---
    info_parser = subparsers.add_parser(
        "info", help="Show details for a specific task."
    )
    info_parser.add_argument("task_name", help="Task name (e.g. 'food_search').")
    info_parser.set_defaults(func=_cmd_info)

    # --- leaderboard ---
    lb_parser = subparsers.add_parser(
        "leaderboard", help="Generate leaderboard from result files."
    )
    lb_parser.add_argument(
        "results_dir", help="Directory containing result JSON files."
    )
    lb_parser.set_defaults(func=_cmd_leaderboard)

    # --- preflight ---
    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Hash benchmark data and scan external files for exact overlap.",
    )
    preflight_parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of tasks to include (default: all).",
    )
    preflight_parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="File or directory to compare against benchmark surfaces. Repeatable.",
    )
    preflight_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the preflight JSON.",
    )
    preflight_parser.set_defaults(func=_cmd_preflight)

    # --- matrix ---
    matrix_parser = subparsers.add_parser(
        "matrix",
        help="Plan or execute the FoodEval baseline matrix.",
    )
    matrix_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON matrix config. Uses the built-in default if omitted.",
    )
    matrix_parser.add_argument(
        "--write-default-config",
        type=str,
        default=None,
        help="Write the built-in matrix JSON to this path and exit.",
    )
    matrix_parser.add_argument(
        "--output-dir",
        type=str,
        default="results/baseline-matrix",
        help="Directory for matrix-plan, preflight, and result JSONs.",
    )
    matrix_parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of tasks to run (default: all).",
    )
    matrix_parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated matrix model IDs to include.",
    )
    matrix_parser.add_argument(
        "--dims",
        type=str,
        default=None,
        help="Comma-separated dimensions to include.",
    )
    matrix_parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="File or directory for contamination preflight. Repeatable.",
    )
    matrix_parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        help="Load KEY=VALUE pairs from an env file before planning/execution. Repeatable.",
    )
    matrix_parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually run evaluations. Omit to only write/print the plan.",
    )
    matrix_parser.add_argument(
        "--fail-missing-env",
        action="store_true",
        default=False,
        help="Attempt runs even if required env vars are absent.",
    )
    matrix_parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List planned model/dimension runs and exit.",
    )
    matrix_parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress per-task progress during --execute.",
    )
    matrix_parser.set_defaults(func=_cmd_matrix)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments. None uses sys.argv.

    Returns:
        Exit code (0 = success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
