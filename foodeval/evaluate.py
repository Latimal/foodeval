"""FoodEval evaluation orchestrator.

Runs one or more benchmark tasks against an embedding adapter and collects
results into a BenchmarkResult. Handles progress output, timing, and
serialization to JSON and markdown.

Usage:
    >>> from foodeval.evaluate import run_benchmark
    >>> from foodeval.adapters.sentence_transformer import SentenceTransformerAdapter
    >>> adapter = SentenceTransformerAdapter("BAAI/bge-m3", truncate_dim=384)  # doctest: +SKIP
    >>> result = run_benchmark(adapter)  # doctest: +SKIP
    >>> print(result.to_markdown())  # doctest: +SKIP
    >>> result.to_json("results/bge-m3-384.json")  # doctest: +SKIP
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from foodeval.adapters.base import EmbeddingAdapter
from foodeval.leaderboard import compute_category_scores
from foodeval.tasks import get_task, list_tasks
from foodeval.tasks.base import TaskResult


@dataclass
class BenchmarkResult:
    """Aggregated results from a full FoodEval evaluation run.

    Attributes:
        model_name: Human-readable model identifier.
        dimension: Embedding dimensionality.
        task_results: Map from task name to TaskResult.
        aggregate_score: Unweighted mean of all task main_scores (1/N per
            task). This is the FoodEval Score.
        category_scores: Mean score per category (Search, Matching,
            Classification), for capability-level comparison only.
        timestamp: ISO-8601 timestamp of the run.
        total_seconds: Total wall-clock time for all tasks.
    """

    model_name: str
    dimension: int
    task_results: dict[str, TaskResult] = field(default_factory=dict)
    aggregate_score: float = 0.0
    category_scores: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""
    total_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "aggregate_score": round(self.aggregate_score, 4),
            "category_scores": {
                k: round(v, 4) for k, v in self.category_scores.items()
            },
            "timestamp": self.timestamp,
            "total_seconds": round(self.total_seconds, 3),
            "metadata": self.metadata,
            "tasks": {name: tr.to_dict() for name, tr in self.task_results.items()},
        }

    def to_json(self, path: str) -> None:
        """Write results to a JSON file.

        Creates parent directories if they don't exist.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def to_markdown(self) -> str:
        """Render a concise markdown summary of all task results."""
        lines: list[str] = []
        lines.append(f"## FoodEval Results: {self.model_name} ({self.dimension}d)")
        lines.append("")
        lines.append(f"Timestamp: {self.timestamp}")
        lines.append(f"Total time: {self.total_seconds:.1f}s")
        lines.append("")

        # Summary table
        lines.append("| Task | Metric | Score |")
        lines.append("|------|--------|------:|")
        for name in sorted(self.task_results):
            tr = self.task_results[name]
            lines.append(f"| {name} | {tr.metric_name} | {tr.main_score:.4f} |")
        if self.category_scores:
            for cat, score in self.category_scores.items():
                lines.append(f"| *{cat} avg* | | *{score:.4f}* |")
        lines.append(f"| **FoodEval Score** | | **{self.aggregate_score:.4f}** |")
        lines.append("")

        # Per-task details
        for name in sorted(self.task_results):
            tr = self.task_results[name]
            lines.append(f"### {name}")
            lines.append(f"- {tr.metric_name}: {tr.main_score:.4f}")
            lines.append(f"- Examples: {tr.n_examples}")
            lines.append(f"- Time: {tr.elapsed_seconds:.2f}s")

            details = tr.details

            # CI if present
            ci = details.get("confidence_interval")
            if ci:
                lines.append(
                    f"- 95% CI: [{ci.get('ci_lower', 0):.4f}, "
                    f"{ci.get('ci_upper', 0):.4f}]"
                )

            # Threshold info for pair classification
            if "best_threshold" in details:
                lines.append(f"- Threshold: {details['best_threshold']:.3f}")
                lines.append(
                    f"- Precision: {details.get('precision', 0):.4f}, "
                    f"Recall: {details.get('recall', 0):.4f}"
                )
                lines.append(f"- AP: {details.get('average_precision', 0):.4f}")

            # Classification variance
            if "std_macro_f1" in details:
                lines.append(
                    f"- Std across {details.get('n_seeds', 0)} seeds: "
                    f"{details['std_macro_f1']:.4f}"
                )

            # Domain breakdown
            per_domain = details.get("per_domain")
            if per_domain:
                lines.append("- Domains:")
                for domain, info in sorted(per_domain.items()):
                    if isinstance(info, dict):
                        score_key = "mean_ndcg" if "mean_ndcg" in info else "best_f1"
                        if score_key in info:
                            lines.append(
                                f"  - {domain}: {info[score_key]:.4f} "
                                f"(n={info.get('n_queries', info.get('n_pairs', '?'))})"
                            )

            lines.append("")

        return "\n".join(lines)


def run_benchmark(
    adapter: EmbeddingAdapter,
    tasks: list[str] | None = None,
    verbose: bool = True,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Run FoodEval evaluation and return aggregated results.

    Args:
        adapter: Embedding adapter to evaluate.
        tasks: List of task names to run. None means all tasks.
        verbose: If True, print progress to stderr.

    Returns:
        BenchmarkResult with per-task scores and aggregate.

    Raises:
        KeyError: If a requested task name is not in the registry.
        FileNotFoundError: If a task's data file is missing.
    """
    if tasks is None:
        task_names = list_tasks()
    else:
        task_names = tasks

    # Validate all task names before starting
    for name in task_names:
        get_task(name)  # raises KeyError if invalid

    from foodeval import __version__

    result = BenchmarkResult(
        model_name=adapter.name,
        dimension=adapter.dimension,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        metadata=metadata or {},
    )
    result.metadata.setdefault("foodeval_version", __version__)

    t0_total = time.monotonic()

    for i, name in enumerate(task_names, 1):
        task = get_task(name)

        if verbose:
            print(
                f"[{i}/{len(task_names)}] Running {name} ({task.task_type})...",
                file=sys.stderr,
                flush=True,
            )

        t0 = time.monotonic()
        try:
            task_result = task.run(adapter)
        except Exception as exc:
            # Isolate a single task's failure so the rest of the suite still
            # runs. KeyboardInterrupt and SystemExit are not Exception
            # subclasses, so they still propagate and abort the run.
            if verbose:
                print(
                    f"  ERROR: {name} failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            task_result = TaskResult(
                task_name=name,
                main_score=0.0,
                metric_name=task.metric_name,
                details={"error": str(exc)},
                n_examples=0,
                elapsed_seconds=time.monotonic() - t0,
                errored=True,
            )

        result.task_results[name] = task_result

        if verbose:
            elapsed = task_result.elapsed_seconds
            print(
                f"  {task_result.metric_name}: {task_result.main_score:.4f} "
                f"({elapsed:.1f}s)",
                file=sys.stderr,
                flush=True,
            )

    result.total_seconds = time.monotonic() - t0_total

    # FoodEval Score = unweighted mean across tasks (1/N each).
    scores = [tr.main_score for tr in result.task_results.values() if not tr.errored]
    result.aggregate_score = round(sum(scores) / len(scores) if scores else 0.0, 4)

    task_score_map = {
        name: tr.main_score
        for name, tr in result.task_results.items()
        if not tr.errored
    }
    result.category_scores = compute_category_scores(task_score_map)

    if verbose:
        print(
            f"\nFoodEval Score: {result.aggregate_score:.4f} "
            f"(total: {result.total_seconds:.1f}s)",
            file=sys.stderr,
            flush=True,
        )

    return result
