"""NDCG@k (Normalized Discounted Cumulative Gain) for ranked retrieval.

Usage:
    >>> from foodeval.metrics.ndcg import ndcg_at_k, mean_ndcg_at_k
    >>> ndcg_at_k([3, 2, 0, 1, 0], k=3)
    0.9467...
    >>> mean_ndcg_at_k([[3, 2, 0], [0, 0, 1]], k=3)
    0.75
"""

from __future__ import annotations

import numpy as np


def _dcg(relevance: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain for the top-k positions."""
    relevance = relevance[:k]
    if len(relevance) == 0:
        return 0.0
    positions = np.arange(1, len(relevance) + 1)
    discounts = np.log2(positions + 1)
    return float(np.sum((2.0**relevance - 1.0) / discounts))


def ndcg_at_k(relevance_scores: list[int], k: int = 10) -> float:
    """Compute NDCG@k for a single ranked list.

    Args:
        relevance_scores: Relevance grades in retrieval order (highest rank first).
        k: Cutoff position. If k > len(relevance_scores), only available
           positions are used.

    Returns:
        NDCG@k in [0, 1]. Returns 0.0 for empty input or all-zero relevance.
    """
    if not relevance_scores:
        return 0.0

    rel = np.asarray(relevance_scores, dtype=np.float64)
    dcg = _dcg(rel, k)

    ideal = np.sort(rel)[::-1]
    idcg = _dcg(ideal, k)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mean_ndcg_at_k(all_relevance: list[list[int]], k: int = 10) -> float:
    """Mean NDCG@k across multiple queries.

    Args:
        all_relevance: List of relevance-score lists, one per query.
        k: Cutoff position.

    Returns:
        Arithmetic mean of per-query NDCG@k. Returns 0.0 for empty input.
    """
    if not all_relevance:
        return 0.0
    scores = [ndcg_at_k(rel, k) for rel in all_relevance]
    return float(np.mean(scores))
