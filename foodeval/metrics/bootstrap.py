"""Bootstrap confidence intervals and paired significance tests.

Usage:
    >>> from foodeval.metrics.bootstrap import bootstrap_ci, bootstrap_paired_test
    >>> ci = bootstrap_ci([0.8, 0.85, 0.9, 0.82, 0.88], seed=42)
    >>> ci["mean"]
    0.85
    >>> test = bootstrap_paired_test(
    ...     [0.8, 0.85, 0.9], [0.7, 0.75, 0.8], seed=42
    ... )
    >>> round(test["mean_diff"], 4)
    0.1
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for the mean of a score distribution.

    Args:
        scores: Observed metric values (one per query/sample).
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95 for 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        Dict with keys: mean, ci_lower, ci_upper, std.
        ``std`` is the standard error of the mean, i.e. the standard deviation
        of the bootstrap distribution of resampled means (not the standard
        deviation of the raw scores). Returns zeros for empty input.
    """
    if not scores:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}

    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)

    if n == 1:
        val = float(arr[0])
        return {"mean": val, "ci_lower": val, "ci_upper": val, "std": 0.0}

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        sample = arr[rng.integers(0, n, size=n)]
        boot_means[i] = sample.mean()

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "mean": float(arr.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
        "std": float(boot_means.std()),
    }


def bootstrap_paired_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap paired permutation test for two models on the same samples.

    Tests whether model A and model B have significantly different means.
    Uses the bootstrap distribution of the paired difference.

    Args:
        scores_a: Per-sample scores from model A.
        scores_b: Per-sample scores from model B (same samples, same order).
        n_bootstrap: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with keys: p_value, mean_diff, ci_lower, ci_upper.
        mean_diff = mean(A) - mean(B). Positive means A is better.
        p_value is two-sided: probability that the observed difference
        is due to chance.
        Returns zeros for empty or mismatched inputs.
    """
    if not scores_a or not scores_b or len(scores_a) != len(scores_b):
        return {"p_value": 1.0, "mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    diffs = a - b
    n = len(diffs)
    observed_diff = float(diffs.mean())

    if n == 1:
        return {
            "p_value": 1.0,
            "mean_diff": observed_diff,
            "ci_lower": observed_diff,
            "ci_upper": observed_diff,
        }

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = diffs[idx].mean()

    # Two-sided p-value: fraction of bootstrap samples where the
    # centered difference is at least as extreme as the observed.
    centered = boot_diffs - boot_diffs.mean()

    # A constant per-sample difference yields zero bootstrap variance: every
    # resample has the identical mean, so the centered distribution is all
    # zeros. The fraction-as-extreme calculation would then read p=0.0 even
    # though there is no evidence about sampling variability. Treat this
    # degenerate case as non-significant.
    if np.allclose(centered, 0.0):
        p_value = 1.0
    else:
        p_value = float(np.mean(np.abs(centered) >= np.abs(observed_diff)))

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    return {
        "p_value": p_value,
        "mean_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
