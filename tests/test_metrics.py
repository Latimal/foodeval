"""Tests for all metric functions: NDCG, F1, classification, and bootstrap.

Each test verifies observable output for given input. Math-heavy tests use
hand-computed expected values or known mathematical identities rather than
reimplementing the formula.
"""

from __future__ import annotations


import numpy as np
import pytest

from foodeval.metrics.ndcg import ndcg_at_k, mean_ndcg_at_k
from foodeval.metrics.f1 import (
    f1_at_threshold,
    best_f1,
    pair_classification_metrics,
    _average_precision,
)
from foodeval.metrics.classification import (
    macro_f1,
    macro_accuracy,
    classification_report,
)
from foodeval.metrics.bootstrap import bootstrap_ci, bootstrap_paired_test


# =========================================================================
# NDCG@k
# =========================================================================


class TestNDCGAtK:
    """NDCG@k: normalized discounted cumulative gain for ranked retrieval."""

    def test_perfect_ranking_gives_one(self):
        """A descending-sorted relevance list is the ideal ranking."""
        assert ndcg_at_k([3, 2, 1, 0], k=10) == 1.0

    def test_already_sorted_single_grade_gives_one(self):
        """When all items share the same nonzero grade, any order is ideal."""
        assert ndcg_at_k([2, 2, 2], k=10) == 1.0

    def test_reverse_ranking_penalized(self):
        """The worst ranking (ascending) scores strictly below 1.0."""
        score = ndcg_at_k([0, 1, 2, 3], k=10)
        assert 0.0 < score < 1.0

    def test_empty_input_returns_zero(self):
        assert ndcg_at_k([], k=10) == 0.0

    def test_all_zeros_returns_zero(self):
        """No relevant documents means NDCG is undefined, returns 0."""
        assert ndcg_at_k([0, 0, 0, 0], k=10) == 0.0

    def test_single_relevant_item_at_top(self):
        """One relevant item at rank 1: NDCG = 1.0."""
        assert ndcg_at_k([3, 0, 0, 0], k=10) == 1.0

    def test_single_relevant_item_at_bottom(self):
        """One relevant item at last rank: NDCG < 1.0."""
        score = ndcg_at_k([0, 0, 0, 3], k=10)
        assert 0.0 < score < 1.0

    def test_k_greater_than_list_length(self):
        """k exceeding input length should use all available positions."""
        score = ndcg_at_k([3, 2], k=100)
        assert score == 1.0

    def test_k_equals_one_only_first_matters(self):
        """With k=1, only the top-ranked item contributes."""
        assert ndcg_at_k([3, 0, 0], k=1) == 1.0
        score_bad = ndcg_at_k([0, 3, 0], k=1)
        assert score_bad == 0.0

    def test_known_value_two_item_swap(self):
        """Hand-computed NDCG@2 for [1, 3] vs ideal [3, 1].

        DCG@2  = (2^1 - 1)/log2(2) + (2^3 - 1)/log2(3)
               = 1/1 + 7/1.5850 = 1.0 + 4.4170 = 5.4170
        IDCG@2 = (2^3 - 1)/log2(2) + (2^1 - 1)/log2(3)
               = 7/1 + 1/1.5850 = 7.0 + 0.6309 = 7.6309
        NDCG@2 = 5.4170 / 7.6309 = 0.7099
        """
        score = ndcg_at_k([1, 3], k=2)
        assert score == pytest.approx(0.7099, abs=0.001)

    def test_k_truncates_relevance_list(self):
        """Items beyond position k should not affect the score.

        [3, 0] at k=1 and [3, 0, 0, 0, 0] at k=1 should both be 1.0,
        and adding a high-relevance item at position 3 should not matter.
        """
        assert ndcg_at_k([3, 0, 0], k=1) == 1.0
        assert ndcg_at_k([3, 0, 3], k=1) == 1.0

    @pytest.mark.parametrize(
        "relevance, k, expected_lower, expected_upper",
        [
            ([3, 0, 2], 3, 0.7, 1.0),  # Slightly suboptimal
            ([1, 1, 1, 1], 4, 0.99, 1.01),  # Uniform = perfect
            ([0, 0, 0, 3], 4, 0.3, 0.6),  # Worst placement
        ],
        ids=["suboptimal", "uniform-relevant", "worst-placement"],
    )
    def test_known_bounds(self, relevance, k, expected_lower, expected_upper):
        score = ndcg_at_k(relevance, k=k)
        assert expected_lower <= score <= expected_upper

    def test_monotonicity_moving_relevant_item_forward(self):
        """Moving a relevant item to an earlier rank should never decrease NDCG."""
        # Relevant item at position 4 vs position 1
        score_late = ndcg_at_k([0, 0, 0, 3], k=4)
        score_early = ndcg_at_k([3, 0, 0, 0], k=4)
        assert score_early >= score_late


class TestMeanNDCGAtK:
    """Mean NDCG@k across multiple queries."""

    def test_empty_list_returns_zero(self):
        assert mean_ndcg_at_k([], k=10) == 0.0

    def test_single_query_equals_ndcg(self):
        """Mean of one query should equal that query's NDCG."""
        single = [3, 2, 1, 0]
        assert mean_ndcg_at_k([single], k=10) == ndcg_at_k(single, k=10)

    def test_two_queries_averaged(self):
        """Mean of a perfect and zero query should be ~0.5."""
        queries = [[3, 2, 1], [0, 0, 0]]
        result = mean_ndcg_at_k(queries, k=10)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_all_perfect_queries_give_one(self):
        queries = [[3, 2, 1], [2, 1], [1]]
        assert mean_ndcg_at_k(queries, k=10) == 1.0

    def test_mean_is_arithmetic_average(self):
        """Verify mean_ndcg is truly the arithmetic mean of individual scores."""
        q1 = [3, 0, 0]
        q2 = [0, 0, 3]
        individual_scores = [ndcg_at_k(q1, k=3), ndcg_at_k(q2, k=3)]
        expected = sum(individual_scores) / len(individual_scores)
        assert mean_ndcg_at_k([q1, q2], k=3) == pytest.approx(expected)


# =========================================================================
# F1 at threshold
# =========================================================================


class TestF1AtThreshold:
    """F1 score with a fixed decision threshold."""

    def test_perfect_separation(self):
        """When scores perfectly separate labels, F1 = 1.0 at the right threshold."""
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]
        result = f1_at_threshold(labels, scores, threshold=0.5)
        assert result["f1"] == 1.0
        assert result["tp"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert result["tn"] == 2

    def test_all_predicted_positive(self):
        """Threshold of 0 predicts everything positive."""
        labels = [1, 0, 1, 0]
        scores = [0.9, 0.8, 0.7, 0.6]
        result = f1_at_threshold(labels, scores, threshold=0.0)
        assert result["tp"] == 2
        assert result["fp"] == 2
        assert result["fn"] == 0
        assert result["tn"] == 0
        assert result["recall"] == 1.0
        assert result["precision"] == 0.5

    def test_all_predicted_negative(self):
        """Threshold above all scores predicts everything negative."""
        labels = [1, 0, 1, 0]
        scores = [0.4, 0.3, 0.2, 0.1]
        result = f1_at_threshold(labels, scores, threshold=0.99)
        assert result["tp"] == 0
        assert result["fn"] == 2
        assert result["f1"] == 0.0

    def test_boundary_score_equals_threshold(self):
        """Scores exactly equal to threshold should be predicted positive."""
        labels = [1, 0]
        scores = [0.5, 0.5]
        result = f1_at_threshold(labels, scores, threshold=0.5)
        assert result["tp"] == 1
        assert result["fp"] == 1

    def test_all_positive_labels(self):
        """When every example is positive, tn=0 and fn depends on threshold."""
        labels = [1, 1, 1]
        scores = [0.9, 0.5, 0.1]
        result = f1_at_threshold(labels, scores, threshold=0.5)
        assert result["tp"] == 2
        assert result["fn"] == 1
        assert result["tn"] == 0
        assert result["fp"] == 0

    def test_all_negative_labels(self):
        """When every example is negative, precision=0 for any positive prediction."""
        labels = [0, 0, 0]
        scores = [0.9, 0.5, 0.1]
        result = f1_at_threshold(labels, scores, threshold=0.5)
        assert result["fp"] == 2
        assert result["tp"] == 0
        assert result["precision"] == 0.0

    def test_return_keys(self):
        """Verify all documented keys are present."""
        result = f1_at_threshold([1, 0], [0.6, 0.4], threshold=0.5)
        expected_keys = {"f1", "precision", "recall", "tp", "fp", "fn", "tn"}
        assert set(result.keys()) == expected_keys

    def test_confusion_counts_sum_to_total(self):
        """tp + fp + fn + tn should always equal the number of examples."""
        labels = [1, 0, 1, 0, 1]
        scores = [0.9, 0.7, 0.4, 0.3, 0.6]
        result = f1_at_threshold(labels, scores, threshold=0.5)
        total = result["tp"] + result["fp"] + result["fn"] + result["tn"]
        assert total == len(labels)

    def test_precision_recall_f1_relationship(self):
        """F1 = 2 * P * R / (P + R), verified by hand.

        labels=[1,1,0,0], scores=[0.9,0.4,0.6,0.1], threshold=0.5
        Preds:  [1,  0,  1,  0]
        TP=1, FP=1, FN=1, TN=1
        P = 1/2 = 0.5, R = 1/2 = 0.5
        F1 = 2*0.5*0.5/(0.5+0.5) = 0.5
        """
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.4, 0.6, 0.1]
        result = f1_at_threshold(labels, scores, threshold=0.5)
        assert result["precision"] == pytest.approx(0.5)
        assert result["recall"] == pytest.approx(0.5)
        assert result["f1"] == pytest.approx(0.5)


class TestBestF1:
    """Threshold-sweeping F1 optimizer."""

    def test_perfect_separation_finds_threshold(self):
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.7, 0.3, 0.1]
        result = best_f1(labels, scores)
        assert result["f1"] == 1.0
        assert 0.3 <= result["threshold"] <= 0.7

    def test_empty_labels_returns_zero(self):
        result = best_f1([], [])
        assert result["f1"] == 0.0
        assert result["threshold"] == 0.5

    def test_identical_scores_handles_gracefully(self):
        """All scores the same: the model has no discriminative power."""
        labels = [1, 0, 1, 0]
        scores = [0.5, 0.5, 0.5, 0.5]
        result = best_f1(labels, scores)
        # Should still produce a valid dict, F1 depends on the threshold
        assert 0.0 <= result["f1"] <= 1.0
        assert "threshold" in result

    def test_custom_thresholds(self):
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.7, 0.3, 0.1]
        result = best_f1(labels, scores, thresholds=[0.5])
        assert result["threshold"] == 0.5
        assert result["f1"] == 1.0

    def test_result_includes_confusion_counts(self):
        result = best_f1([1, 0, 1, 0], [0.8, 0.6, 0.4, 0.2])
        for key in ("tp", "fp", "fn", "tn"):
            assert key in result
            assert isinstance(result[key], int)

    def test_threshold_in_result_dict(self):
        result = best_f1([1, 0], [0.8, 0.2])
        assert "threshold" in result

    def test_best_f1_is_at_least_as_good_as_midpoint(self):
        """The optimized threshold should never be worse than 0.5."""
        labels = [1, 1, 0, 0, 1, 0]
        scores = [0.95, 0.85, 0.15, 0.05, 0.55, 0.45]
        best = best_f1(labels, scores)
        mid = f1_at_threshold(labels, scores, threshold=0.5)
        assert best["f1"] >= mid["f1"]

    def test_all_positive_labels_finds_low_threshold(self):
        """When all labels are 1, the best threshold predicts all positive."""
        labels = [1, 1, 1, 1]
        scores = [0.9, 0.7, 0.3, 0.1]
        result = best_f1(labels, scores)
        # F1=1.0 achievable by predicting all positive
        assert result["f1"] == 1.0
        assert result["threshold"] <= 0.1

    def test_negative_cosine_similarities(self):
        """Cosine similarity ranges [-1, 1]. Threshold sweep must handle negatives."""
        labels = [1, 1, 0, 0]
        scores = [-0.1, -0.3, -0.7, -0.9]
        # Use score values as thresholds to test negative range
        thresholds = sorted(set(scores))
        result = best_f1(labels, scores, thresholds=thresholds)
        assert result["f1"] == 1.0
        assert result["threshold"] < 0.0

    def test_empty_thresholds_list(self):
        """An empty thresholds list should raise ValueError."""
        labels = [1, 0, 1, 0]
        scores = [0.9, 0.7, 0.3, 0.1]
        with pytest.raises(ValueError, match="thresholds must not be empty"):
            best_f1(labels, scores, thresholds=[])


class TestAveragePrecision:
    """Internal _average_precision function."""

    def test_perfect_ranking_gives_one(self):
        """All positives ranked before negatives."""
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]
        assert _average_precision(labels, scores) == 1.0

    def test_worst_ranking(self):
        """All negatives ranked before positives."""
        labels = [0, 0, 1, 1]
        scores = [0.9, 0.8, 0.2, 0.1]
        ap = _average_precision(labels, scores)
        assert ap < 0.5

    def test_empty_input_returns_zero(self):
        assert _average_precision([], []) == 0.0

    def test_no_positives_returns_zero(self):
        assert _average_precision([0, 0, 0], [0.9, 0.5, 0.1]) == 0.0

    def test_known_value(self):
        """Hand-computed AP for [1,0,1,0] with scores [0.9,0.8,0.3,0.1].

        Sorted by score desc: [1,0,1,0]
        Position 1: pos, P@1=1/1=1.0
        Position 2: neg, skip
        Position 3: pos, P@3=2/3=0.667
        AP = (1.0 + 0.667) / 2 = 0.8333
        """
        labels = [1, 0, 1, 0]
        scores = [0.9, 0.8, 0.3, 0.1]
        assert _average_precision(labels, scores) == pytest.approx(0.8333, abs=0.001)

    def test_all_scores_tied_equals_positive_base_rate(self):
        """All scores tied: sklearn's AP collapses to the positive base rate.

        With every score identical there is no rank information, so AP equals
        n_positive / n_total. Here 2 positives out of 4 => 0.5. Pinned to the
        sklearn reference value (best_f1 sweeps thresholds, but AP does not).
        """
        assert _average_precision([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(
            0.5
        )

    def test_partial_tie_pins_sklearn_value(self):
        """Two positives and one negative tied at the top, one negative below.

        Sorted desc the tie block {0.8,0.8,0.8} mixes 2 pos + 1 neg above a
        lone negative. sklearn's average_precision_score yields exactly 2/3 for
        this configuration; pin it so a change in tie handling is caught.
        """
        ap = _average_precision([1, 1, 0, 0], [0.8, 0.8, 0.8, 0.2])
        assert ap == pytest.approx(0.6667, abs=1e-4)


class TestPairClassificationMetrics:
    """Combined pair classification metrics (F1 + average precision)."""

    def test_perfect_separation(self):
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]
        result = pair_classification_metrics(labels, scores)
        assert result["best_f1"] == 1.0
        assert result["average_precision"] == 1.0

    def test_return_keys(self):
        result = pair_classification_metrics([1, 0], [0.6, 0.4])
        expected_keys = {"best_f1", "best_threshold", "average_precision", "max_ap"}
        assert set(result.keys()) == expected_keys

    def test_all_same_label_positive(self):
        """All positives: AP should be 1.0 since every prediction is correct."""
        labels = [1, 1, 1]
        scores = [0.9, 0.5, 0.1]
        result = pair_classification_metrics(labels, scores)
        assert result["average_precision"] == 1.0

    def test_inverted_scores_penalizes_ap(self):
        """When high scores correspond to negatives, AP is poor."""
        labels = [0, 0, 1, 1]
        scores = [0.9, 0.8, 0.2, 0.1]
        result = pair_classification_metrics(labels, scores)
        assert result["average_precision"] < 0.5

    def test_max_ap_equals_average_precision(self):
        """max_ap should equal average_precision in the current implementation."""
        labels = [1, 0, 1, 0]
        scores = [0.8, 0.6, 0.4, 0.2]
        result = pair_classification_metrics(labels, scores)
        assert result["max_ap"] == result["average_precision"]

    def test_best_threshold_is_in_unit_interval(self):
        result = pair_classification_metrics(
            [1, 0, 1, 0, 0, 1],
            [0.95, 0.65, 0.55, 0.35, 0.15, 0.75],
        )
        assert 0.0 <= result["best_threshold"] <= 1.0

    def test_tied_scores_delegates_ap_to_sklearn(self):
        """With all scores tied, AP equals the sklearn base-rate value (0.5),
        while best_f1 still finds the all-positive operating point (F1=2/3).

        Confirms pair_classification_metrics forwards ties to sklearn's AP
        rather than reimplementing the precision-recall integral.
        """
        result = pair_classification_metrics([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
        assert result["average_precision"] == pytest.approx(0.5)
        assert result["max_ap"] == pytest.approx(0.5)
        # Single threshold (0.5) predicts all positive: P=0.5, R=1.0 -> F1=2/3.
        assert result["best_f1"] == pytest.approx(0.6667, abs=1e-4)


# =========================================================================
# Classification metrics
# =========================================================================


class TestMacroF1:
    """Macro-averaged F1 score."""

    def test_perfect_predictions(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 2, 2]
        assert macro_f1(y_true, y_pred) == 1.0

    def test_completely_wrong(self):
        """Every prediction is wrong: macro F1 should be 0.0."""
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        assert macro_f1(y_true, y_pred) == 0.0

    def test_empty_inputs(self):
        assert macro_f1([], []) == 0.0

    def test_single_class_perfect(self):
        assert macro_f1([0, 0, 0], [0, 0, 0]) == 1.0

    def test_partial_correctness(self):
        """Half correct for each of 2 classes: F1 should be between 0 and 1."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]
        f1 = macro_f1(y_true, y_pred)
        assert 0.0 < f1 < 1.0

    def test_known_value_three_classes(self):
        """Hand-verifiable: class 1 perfect, class 0 diluted, class 2 missed.

        Class 0: TP=2, FP=2 (class 2 items predicted as 0). P=2/4=0.5, R=2/2=1.0, F1=0.667
        Class 1: TP=2, FP=0, FN=0. P=1.0, R=1.0, F1=1.0
        Class 2: TP=0, FN=2. P=0, R=0, F1=0
        Macro = (0.667 + 1.0 + 0.0) / 3 = 0.5556
        """
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 0, 0]
        assert macro_f1(y_true, y_pred) == pytest.approx(0.5556, abs=0.001)


class TestMacroAccuracy:
    """Macro-averaged per-class accuracy."""

    def test_perfect_predictions(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 2, 2]
        assert macro_accuracy(y_true, y_pred) == 1.0

    def test_empty_inputs(self):
        assert macro_accuracy([], []) == 0.0

    def test_imbalanced_accuracy(self):
        """Macro accuracy averages per-class, not globally.

        Class 0: 2/2 correct = 1.0
        Class 1: 0/2 correct = 0.0
        Macro = 0.5
        """
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 0, 0]
        assert macro_accuracy(y_true, y_pred) == pytest.approx(0.5)

    def test_symmetry_with_two_classes(self):
        """Swapping true/pred labels should maintain the same macro accuracy
        when each class has the same support."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]
        assert macro_accuracy(y_true, y_pred) == pytest.approx(0.5)

    def test_all_wrong_predictions(self):
        """Every prediction is wrong for each class."""
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        assert macro_accuracy(y_true, y_pred) == pytest.approx(0.0)


class TestClassificationReport:
    """Per-class classification report."""

    def test_report_structure(self):
        report = classification_report([0, 1, 1], [0, 1, 0])
        assert "macro_f1" in report
        assert "macro_accuracy" in report
        assert "per_class" in report

    def test_per_class_keys(self):
        report = classification_report([0, 1], [0, 1], label_names=["cat", "dog"])
        assert "cat" in report["per_class"]
        assert "dog" in report["per_class"]
        for cls_report in report["per_class"].values():
            assert set(cls_report.keys()) == {"precision", "recall", "f1", "support"}

    def test_perfect_report(self):
        report = classification_report([0, 0, 1, 1], [0, 0, 1, 1])
        assert report["macro_f1"] == 1.0
        for cls_report in report["per_class"].values():
            assert cls_report["f1"] == 1.0

    def test_empty_returns_zeros(self):
        report = classification_report([], [])
        assert report["macro_f1"] == 0.0
        assert report["per_class"] == {}

    def test_label_names_mapping(self):
        """label_names should map integer labels to human-readable strings."""
        report = classification_report(
            [0, 1, 2],
            [0, 1, 2],
            label_names=["Indian", "Italian", "Thai"],
        )
        assert "Indian" in report["per_class"]
        assert "Italian" in report["per_class"]
        assert "Thai" in report["per_class"]

    def test_support_counts_match_input(self):
        """Support for each class should equal its count in y_true."""
        report = classification_report(
            [0, 0, 0, 1, 1, 2],
            [0, 0, 1, 1, 1, 2],
            label_names=["a", "b", "c"],
        )
        assert report["per_class"]["a"]["support"] == 3
        assert report["per_class"]["b"]["support"] == 2
        assert report["per_class"]["c"]["support"] == 1

    def test_without_label_names_uses_integers(self):
        """When label_names is None, class names should be string integers."""
        report = classification_report([0, 1], [0, 1])
        assert "0" in report["per_class"]
        assert "1" in report["per_class"]


# =========================================================================
# Bootstrap CI and paired test
# =========================================================================


class TestBootstrapCI:
    """Bootstrap confidence interval for the mean."""

    def test_deterministic_with_seed(self):
        """Same inputs and seed should produce identical results."""
        scores = [0.8, 0.85, 0.9, 0.82, 0.88]
        r1 = bootstrap_ci(scores, seed=42)
        r2 = bootstrap_ci(scores, seed=42)
        assert r1 == r2

    def test_different_seeds_differ(self):
        scores = [0.8, 0.85, 0.9, 0.82, 0.88]
        r1 = bootstrap_ci(scores, seed=42)
        r2 = bootstrap_ci(scores, seed=99)
        # Means are the same (data is the same), but CI bounds differ
        assert r1["mean"] == r2["mean"]
        assert r1["ci_lower"] != r2["ci_lower"]

    def test_mean_is_sample_mean(self):
        scores = [0.8, 0.85, 0.9, 0.82, 0.88]
        result = bootstrap_ci(scores, seed=42)
        assert result["mean"] == pytest.approx(np.mean(scores))

    def test_ci_contains_mean(self):
        """The confidence interval should bracket the sample mean."""
        scores = [0.7, 0.75, 0.8, 0.85, 0.9, 0.78, 0.82]
        result = bootstrap_ci(scores, seed=42)
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_empty_returns_zeros(self):
        result = bootstrap_ci([])
        assert result == {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}

    def test_single_value(self):
        result = bootstrap_ci([0.85])
        assert result["mean"] == 0.85
        assert result["ci_lower"] == 0.85
        assert result["ci_upper"] == 0.85
        assert result["std"] == 0.0

    def test_return_keys(self):
        result = bootstrap_ci([0.5, 0.6, 0.7])
        assert set(result.keys()) == {"mean", "ci_lower", "ci_upper", "std"}

    def test_wider_ci_with_higher_variance(self):
        """Scores with more spread should produce a wider CI."""
        tight = bootstrap_ci([0.80, 0.81, 0.82, 0.79, 0.80], seed=42)
        wide = bootstrap_ci([0.50, 0.60, 0.70, 0.90, 1.00], seed=42)
        tight_width = tight["ci_upper"] - tight["ci_lower"]
        wide_width = wide["ci_upper"] - wide["ci_lower"]
        assert wide_width > tight_width

    def test_ci_level_affects_width(self):
        """A 99% CI should be wider than a 90% CI."""
        scores = [0.7, 0.75, 0.8, 0.85, 0.9, 0.78, 0.82]
        ci_90 = bootstrap_ci(scores, ci=0.90, seed=42)
        ci_99 = bootstrap_ci(scores, ci=0.99, seed=42)
        width_90 = ci_90["ci_upper"] - ci_90["ci_lower"]
        width_99 = ci_99["ci_upper"] - ci_99["ci_lower"]
        assert width_99 > width_90

    def test_all_identical_scores(self):
        """When every score is the same, CI collapses to a point."""
        result = bootstrap_ci([0.85, 0.85, 0.85, 0.85], seed=42)
        assert result["mean"] == 0.85
        assert result["ci_lower"] == pytest.approx(0.85)
        assert result["ci_upper"] == pytest.approx(0.85)
        assert result["std"] == pytest.approx(0.0, abs=1e-10)

    def test_std_is_nonnegative(self):
        result = bootstrap_ci([0.5, 0.6, 0.7, 0.8, 0.9])
        assert result["std"] >= 0.0

    def test_lower_never_exceeds_upper(self):
        result = bootstrap_ci([0.1, 0.5, 0.9, 0.3, 0.7])
        assert result["ci_lower"] <= result["ci_upper"]


class TestBootstrapPairedTest:
    """Bootstrap paired significance test."""

    def test_identical_scores_high_p_value(self):
        """Identical score vectors: no significant difference, p near 1.0."""
        scores = [0.8, 0.85, 0.9, 0.82, 0.88]
        result = bootstrap_paired_test(scores, scores, seed=42)
        assert result["mean_diff"] == 0.0
        assert result["p_value"] >= 0.5  # Not significant

    def test_clearly_different_models(self):
        """Model A consistently outperforms model B: p should be small.

        Per-sample differences vary (non-degenerate bootstrap variance) so the
        test exercises the normal significance path rather than the constant
        difference guard.
        """
        a = [0.90, 0.92, 0.85, 0.95, 0.88, 0.93, 0.87, 0.91]
        b = [0.70, 0.75, 0.68, 0.72, 0.66, 0.74, 0.69, 0.71]
        result = bootstrap_paired_test(a, b, seed=42)
        assert result["mean_diff"] > 0.0
        assert result["p_value"] < 0.05

    def test_constant_difference_returns_p_one(self):
        """A perfectly constant per-sample difference has zero bootstrap
        variance, so the test cannot establish significance and returns p=1.0."""
        a = [0.90, 0.92, 0.91, 0.93, 0.89]
        b = [0.70, 0.72, 0.71, 0.73, 0.69]
        result = bootstrap_paired_test(a, b, seed=42)
        assert result["mean_diff"] == pytest.approx(0.2)
        assert result["p_value"] == 1.0

    def test_mean_diff_is_a_minus_b(self):
        a = [0.9, 0.8]
        b = [0.7, 0.6]
        result = bootstrap_paired_test(a, b, seed=42)
        assert result["mean_diff"] == pytest.approx(0.2)

    def test_negative_mean_diff_when_b_better(self):
        """When B > A, mean_diff should be negative."""
        a = [0.5, 0.6]
        b = [0.9, 0.8]
        result = bootstrap_paired_test(a, b, seed=42)
        assert result["mean_diff"] < 0.0

    def test_empty_returns_defaults(self):
        result = bootstrap_paired_test([], [], seed=42)
        assert result["p_value"] == 1.0
        assert result["mean_diff"] == 0.0

    def test_mismatched_lengths_returns_defaults(self):
        result = bootstrap_paired_test([0.8, 0.9], [0.7], seed=42)
        assert result["p_value"] == 1.0

    def test_return_keys(self):
        result = bootstrap_paired_test([0.5], [0.6], seed=42)
        assert set(result.keys()) == {"p_value", "mean_diff", "ci_lower", "ci_upper"}

    def test_ci_bracket_for_nonzero_diff(self):
        """When A is always better, the CI should not contain 0."""
        a = [0.95, 0.96, 0.94, 0.93, 0.95, 0.94, 0.96, 0.95]
        b = [0.70, 0.71, 0.69, 0.68, 0.70, 0.69, 0.71, 0.70]
        result = bootstrap_paired_test(a, b, n_bootstrap=5000, seed=42)
        assert result["ci_lower"] > 0.0

    def test_deterministic_with_seed(self):
        a = [0.8, 0.85, 0.9]
        b = [0.7, 0.75, 0.8]
        r1 = bootstrap_paired_test(a, b, seed=42)
        r2 = bootstrap_paired_test(a, b, seed=42)
        assert r1 == r2

    def test_single_sample_returns_p_one(self):
        """Can't do significance testing with n=1."""
        result = bootstrap_paired_test([0.9], [0.7], seed=42)
        assert result["p_value"] == 1.0
        assert result["mean_diff"] == pytest.approx(0.2)

    def test_p_value_in_unit_interval(self):
        """P-value should always be between 0 and 1."""
        a = [0.8, 0.85, 0.9, 0.82, 0.88]
        b = [0.75, 0.80, 0.85, 0.77, 0.83]
        result = bootstrap_paired_test(a, b, seed=42)
        assert 0.0 <= result["p_value"] <= 1.0
