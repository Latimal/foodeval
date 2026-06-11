"""Pair classification task: cosine similarity + threshold sweep for best F1.

Handles pair-matching tasks (indian_match, global_match, beverage_match,
bakery_match, portion_size, noisy_menu_match) and cross_lingual_match. All share the same
data format and evaluation logic: encode text_a and text_b, compute cosine
similarity, sweep thresholds to find the one that maximizes F1.

Usage:
    >>> from foodeval.tasks.pair_classification import PairClassificationTask
    >>> task = PairClassificationTask("indian_match")
    >>> result = task.run(adapter)  # doctest: +SKIP
    >>> print(f"best_f1: {result.main_score:.4f}")  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any

import numpy as np

from foodeval.adapters.base import EmbeddingAdapter
from foodeval.metrics import best_f1, pair_classification_metrics
from foodeval.metrics.f1 import f1_at_threshold
from foodeval.tasks.base import BenchmarkTask, TaskResult


class PairClassificationTask(BenchmarkTask):
    """Pair classification via cosine similarity and threshold-optimized F1.

    For each pair, cosine similarity between embeddings of text_a and text_b
    is computed. A threshold sweep finds the operating point that maximizes
    F1 score. Average precision is also reported.
    """

    task_type = "pair_classification"
    metric_name = "best_f1"

    def __init__(self, task_name: str) -> None:
        super().__init__(task_name)
        self._pairs: list[dict[str, Any]] = []

    def load_data(self) -> None:
        """Load pairs from the task JSON file.

        Validates that each pair has text_a, text_b, and a binary label.
        """
        data = self._load_json()
        self._data = data

        self._pairs = data["pairs"]

        if not self._pairs:
            raise ValueError(f"{self.name}: no pairs found")

        for p in self._pairs:
            if "text_a" not in p or "text_b" not in p:
                raise ValueError(
                    f"{self.name}: pair {p.get('id', '?')} missing text_a or text_b"
                )
            if p.get("label") not in (0, 1):
                raise ValueError(
                    f"{self.name}: pair {p.get('id', '?')} has invalid label "
                    f"{p.get('label')!r} (expected 0 or 1)"
                )

    @staticmethod
    def _bootstrap_f1_ci(
        labels: list[int],
        similarities: list[float],
        threshold: float,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        seed: int = 42,
    ) -> dict:
        """Bootstrap confidence interval over F1 at a fixed threshold.

        For each resample, computes F1 from the resampled pairs. The CI
        brackets the reported F1 metric, not per-pair accuracy.
        """
        n = len(labels)
        if n == 0:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}

        labels_arr = np.asarray(labels, dtype=np.int64)
        sims_arr = np.asarray(similarities, dtype=np.float64)
        rng = np.random.default_rng(seed)

        boot_f1s = np.empty(n_bootstrap, dtype=np.float64)
        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            result = f1_at_threshold(
                labels_arr[idx].tolist(), sims_arr[idx].tolist(), threshold
            )
            boot_f1s[i] = result["f1"]

        alpha = 1.0 - ci
        return {
            "mean": float(np.mean(boot_f1s)),
            "ci_lower": float(np.percentile(boot_f1s, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(boot_f1s, 100 * (1 - alpha / 2))),
            "std": float(np.std(boot_f1s)),
        }

    def evaluate(self, adapter: EmbeddingAdapter) -> TaskResult:
        """Encode pairs, compute cosine similarity, find best F1 threshold.

        Returns:
            TaskResult with best_f1, plus precision, recall, threshold,
            average precision, per-domain breakdown, and bootstrap CI.
        """
        if not self._pairs:
            raise RuntimeError(f"{self.name}: call load_data() before evaluate()")

        texts_a = [p["text_a"] for p in self._pairs]
        texts_b = [p["text_b"] for p in self._pairs]
        labels = [p["label"] for p in self._pairs]

        # Encode both sides
        emb_a = adapter.encode(texts_a, normalize=True)
        emb_b = adapter.encode(texts_b, normalize=True)

        # Cosine similarity (already normalized)
        similarities = np.sum(emb_a * emb_b, axis=1).tolist()

        # Overall metrics
        overall = pair_classification_metrics(labels, similarities)
        bf = best_f1(labels, similarities)

        # Bootstrap CI over F1 at the best threshold: resample pairs,
        # compute F1 for each bootstrap sample, report CI over those F1s.
        ci = self._bootstrap_f1_ci(labels, similarities, bf["threshold"])

        # Per-domain breakdown
        per_domain: dict[str, dict[str, list]] = {}
        for i, p in enumerate(self._pairs):
            domain = p.get("domain", "unknown")
            per_domain.setdefault(domain, {"labels": [], "sims": []})
            per_domain[domain]["labels"].append(labels[i])
            per_domain[domain]["sims"].append(similarities[i])

        domain_summary: dict[str, dict[str, Any]] = {}
        for domain, group in sorted(per_domain.items()):
            domain_bf = best_f1(group["labels"], group["sims"])
            domain_summary[domain] = {
                "best_f1": round(domain_bf["f1"], 4),
                "threshold": round(domain_bf["threshold"], 3),
                "n_pairs": len(group["labels"]),
                "n_positive": sum(group["labels"]),
                "n_negative": len(group["labels"]) - sum(group["labels"]),
            }

        predictions = [
            {
                "id": p.get("id", f"{self.name}_{i}"),
                "domain": p.get("domain", "unknown"),
                "label": labels[i],
                "score": round(float(similarities[i]), 6),
                "predicted": int(similarities[i] >= bf["threshold"]),
            }
            for i, p in enumerate(self._pairs)
        ]

        return TaskResult(
            task_name=self.name,
            main_score=round(bf["f1"], 4),
            metric_name=self.metric_name,
            n_examples=len(self._pairs),
            details={
                "best_threshold": round(bf["threshold"], 3),
                "precision": round(bf["precision"], 4),
                "recall": round(bf["recall"], 4),
                "average_precision": round(overall["average_precision"], 4),
                "tp": bf["tp"],
                "fp": bf["fp"],
                "fn": bf["fn"],
                "tn": bf["tn"],
                "per_domain": domain_summary,
                "confidence_interval": ci,
                "predictions": predictions,
            },
        )
