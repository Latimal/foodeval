"""Abstract base class for all FoodEval tasks.

Every task (retrieval, pair classification, classification) inherits from
BenchmarkTask and implements load_data() and evaluate(). Results are returned
as TaskResult dataclasses, which carry the primary metric, per-domain breakdowns,
confidence intervals, and raw counts.

Usage:
    >>> from foodeval.tasks.base import BenchmarkTask, TaskResult
    >>> class MyTask(BenchmarkTask):
    ...     def load_data(self) -> None: ...
    ...     def evaluate(self, adapter) -> TaskResult: ...
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from foodeval.adapters.base import EmbeddingAdapter

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class TaskResult:
    """Container for a single task's evaluation output.

    Attributes:
        task_name: Identifier matching the task's registry key.
        main_score: The task's primary metric value (e.g. NDCG@10, best F1).
        metric_name: Human-readable metric label (e.g. "ndcg@10", "best_f1").
        details: Arbitrary dict with per-domain scores, CIs, thresholds, etc.
        n_examples: Number of evaluation examples (queries, pairs, or items).
        elapsed_seconds: Wall-clock time for evaluation.
    """

    task_name: str
    main_score: float
    metric_name: str
    details: dict[str, Any] = field(default_factory=dict)
    n_examples: int = 0
    elapsed_seconds: float = 0.0
    errored: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d = {
            "task_name": self.task_name,
            "main_score": self.main_score,
            "metric_name": self.metric_name,
            "details": self.details,
            "n_examples": self.n_examples,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }
        if self.errored:
            d["errored"] = True
        return d


class BenchmarkTask(ABC):
    """Abstract base for all FoodEval evaluation tasks.

    Subclasses set ``name``, ``task_type``, and ``metric_name`` as class
    attributes, then implement ``load_data`` and ``evaluate``.
    """

    name: str
    task_type: str  # "retrieval", "pair_classification", "classification"
    metric_name: str

    def __init__(self, task_name: str) -> None:
        self.name = task_name
        self._data: dict[str, Any] | None = None
        self._data_path = DATA_DIR / f"{task_name}.json"

    @property
    def data_path(self) -> Path:
        """Path to the task's JSON data file."""
        return self._data_path

    def _load_json(self) -> dict[str, Any]:
        """Read and parse the task JSON file.

        Raises:
            FileNotFoundError: If the data file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        if not self._data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self._data_path}. "
                f"Ensure the {self.name} benchmark data is installed in "
                f"{DATA_DIR}/"
            )
        with open(self._data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @abstractmethod
    def load_data(self) -> None:
        """Load and validate benchmark data from disk.

        Called once before evaluate(). Implementations should populate
        internal state and raise on malformed data.
        """
        ...

    @abstractmethod
    def evaluate(self, adapter: EmbeddingAdapter) -> TaskResult:
        """Run evaluation and return results.

        Args:
            adapter: An embedding adapter satisfying the EmbeddingAdapter protocol.

        Returns:
            TaskResult with the primary metric, details, and timing.
        """
        ...

    def run(self, adapter: EmbeddingAdapter) -> TaskResult:
        """Load data (if needed) and evaluate. Measures wall-clock time.

        This is the primary entry point for the evaluation harness.
        """
        if self._data is None:
            self.load_data()

        t0 = time.monotonic()
        result = self.evaluate(adapter)
        result.elapsed_seconds = time.monotonic() - t0
        return result

    def describe(self) -> dict[str, Any]:
        """Return a human-readable description of the task.

        Loads data if not already loaded, so metadata is available.
        """
        if self._data is None:
            self.load_data()

        info: dict[str, Any] = {
            "name": self.name,
            "task_type": self.task_type,
            "metric": self.metric_name,
        }
        if self._data is not None:
            info["version"] = self._data.get("version", "unknown")
            info["metadata"] = self._data.get("metadata", {})
        return info

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
