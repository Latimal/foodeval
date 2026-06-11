"""F1 score with threshold sweeping for pair classification tasks.

Usage:
    >>> from foodeval.metrics.f1 import best_f1, pair_classification_metrics
    >>> result = best_f1([1, 1, 0, 0], [0.9, 0.7, 0.3, 0.1])
    >>> result["f1"]
    1.0
    >>> metrics = pair_classification_metrics([1, 1, 0, 0], [0.9, 0.7, 0.3, 0.1])
    >>> metrics["best_f1"]
    1.0
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def f1_at_threshold(labels: list[int], scores: list[float], threshold: float) -> dict:
    """Compute F1, precision, recall, and confusion counts at a fixed threshold.

    Args:
        labels: Binary ground-truth labels (0 or 1).
        scores: Predicted similarity scores.
        threshold: Decision boundary. Scores >= threshold are predicted positive.

    Returns:
        Dict with keys: f1, precision, recall, tp, fp, fn, tn.
    """
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)

    preds = (s >= threshold).astype(np.int64)

    tp = int(np.sum((preds == 1) & (y == 1)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    tn = int(np.sum((preds == 0) & (y == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def best_f1(
    labels: list[int],
    scores: list[float],
    thresholds: list[float] | None = None,
) -> dict:
    """Sweep thresholds and return the one that maximizes F1.

    Args:
        labels: Binary ground-truth labels (0 or 1).
        scores: Predicted similarity scores.
        thresholds: Candidate thresholds to evaluate. If None, the sorted
            unique observed scores are used, so the true optimum is found.

    Returns:
        Dict with keys: f1, precision, recall, threshold, tp, fp, fn, tn.
        Returns all-zero metrics with threshold=0.5 for empty inputs.
    """
    if not labels or not scores:
        return {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "threshold": 0.5,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    if thresholds is None:
        unique_scores = sorted(set(scores))
        if unique_scores:
            thresholds = unique_scores
        else:
            thresholds = np.linspace(0.0, 1.0, 201).tolist()

    if not thresholds:
        raise ValueError("thresholds must not be empty")

    best_result: dict | None = None
    best_score = -1.0

    for t in thresholds:
        result = f1_at_threshold(labels, scores, t)
        if result["f1"] > best_score:
            best_score = result["f1"]
            best_result = {**result, "threshold": t}

    assert best_result is not None
    return best_result


def _average_precision(labels: list[int], scores: list[float]) -> float:
    """Average precision (area under the precision-recall curve).

    Delegates to ``sklearn.metrics.average_precision_score`` so tied scores are
    handled identically to the reference implementation. Returns 0.0 when there
    are no examples or no positives.
    """
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)

    if len(y) == 0 or np.sum(y) == 0:
        return 0.0

    return float(average_precision_score(y, s))


def pair_classification_metrics(labels: list[int], scores: list[float]) -> dict:
    """Compute pair classification metrics: best F1 and average precision.

    Args:
        labels: Binary ground-truth labels (0 or 1).
        scores: Predicted similarity scores.

    Returns:
        Dict with keys: best_f1, best_threshold, average_precision, max_ap.
    """
    bf = best_f1(labels, scores)
    ap = _average_precision(labels, scores)

    return {
        "best_f1": bf["f1"],
        "best_threshold": bf["threshold"],
        "average_precision": ap,
        "max_ap": ap,
    }
