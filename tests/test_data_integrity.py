"""Data integrity tests for all FoodEval benchmark files.

These tests validate the structure, consistency, and size requirements of
every JSON data file shipped with the package. They catch issues like:
- Malformed JSON
- Missing required fields
- Duplicate IDs
- Label/relevance values out of range
- Corpus references that point to nonexistent items
- Metadata counts that disagree with actual data
- Files that are too small to be meaningful benchmarks
- Missing domain annotations
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from foodeval.tasks.base import DATA_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_data_file(name: str) -> dict:
    """Load a benchmark data file and return parsed JSON."""
    path = DATA_DIR / f"{name}.json"
    assert path.exists(), f"Data file missing: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# All data files that should exist
SEARCH_TASKS = ["food_search", "concept_search", "diet_search", "noisy_search"]
PAIR_TASKS = [
    "indian_match",
    "global_match",
    "beverage_match",
    "bakery_match",
    "portion_size",
    "noisy_menu_match",
    "cross_lingual_match",
]
CLASSIFICATION_TASKS = ["cuisine_classify"]
ALL_TASKS = SEARCH_TASKS + PAIR_TASKS + CLASSIFICATION_TASKS


# =========================================================================
# File existence and valid JSON
# =========================================================================


class TestDataFilesExist:
    """Every expected data file must exist and be valid JSON."""

    @pytest.mark.parametrize("task_name", ALL_TASKS)
    def test_file_exists(self, task_name):
        path = DATA_DIR / f"{task_name}.json"
        assert path.exists(), f"Expected data file not found: {path}"

    @pytest.mark.parametrize("task_name", ALL_TASKS)
    def test_file_is_valid_json(self, task_name):
        path = DATA_DIR / f"{task_name}.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)  # Raises JSONDecodeError if invalid
        assert isinstance(data, dict)

    @pytest.mark.parametrize("task_name", ALL_TASKS)
    def test_file_has_version_field(self, task_name):
        data = _load_data_file(task_name)
        assert "version" in data, f"{task_name}: missing 'version' field"

    @pytest.mark.parametrize("task_name", ALL_TASKS)
    def test_file_has_description(self, task_name):
        data = _load_data_file(task_name)
        assert "description" in data, f"{task_name}: missing 'description' field"
        assert len(data["description"]) > 0


# =========================================================================
# Search task data integrity
# =========================================================================


class TestSearchTaskData:
    """Structure and content validation for retrieval/search tasks."""

    @pytest.mark.parametrize("task_name", SEARCH_TASKS)
    def test_required_top_level_keys(self, task_name):
        data = _load_data_file(task_name)
        for key in ("task", "version", "description", "metric", "corpus", "queries"):
            assert key in data, f"{task_name}: missing top-level key '{key}'"

    @pytest.mark.parametrize("task_name", SEARCH_TASKS)
    def test_corpus_nonempty(self, task_name):
        data = _load_data_file(task_name)
        assert len(data["corpus"]) >= 10, (
            f"{task_name}: corpus has only {len(data['corpus'])} items"
        )

    @pytest.mark.parametrize("task_name", SEARCH_TASKS)
    def test_minimum_query_count(self, task_name):
        """Each search task should have at least 20 queries."""
        data = _load_data_file(task_name)
        assert len(data["queries"]) >= 20, (
            f"{task_name}: only {len(data['queries'])} queries (minimum 20)"
        )

    @pytest.mark.parametrize("task_name", SEARCH_TASKS)
    def test_queries_have_required_fields(self, task_name):
        data = _load_data_file(task_name)
        for q in data["queries"]:
            assert "id" in q, f"{task_name}: query missing 'id'"
            assert "query" in q, (
                f"{task_name}: query {q.get('id', '?')} missing 'query'"
            )
            assert "relevance" in q, (
                f"{task_name}: query {q.get('id', '?')} missing 'relevance'"
            )

    @pytest.mark.parametrize("task_name", SEARCH_TASKS)
    def test_no_duplicate_query_ids(self, task_name):
        data = _load_data_file(task_name)
        ids = [q["id"] for q in data["queries"]]
        assert len(ids) == len(set(ids)), f"{task_name}: duplicate query IDs found"

    @pytest.mark.parametrize("task_name", SEARCH_TASKS)
    def test_relevance_grades_valid(self, task_name):
        """Relevance grades must be integers in {0, 1, 2, 3}."""
        data = _load_data_file(task_name)
        valid_grades = {0, 1, 2, 3}
        for q in data["queries"]:
            for item, grade in q["relevance"].items():
                assert grade in valid_grades, (
                    f"{task_name}: query {q['id']} has invalid grade {grade} "
                    f"for item '{item}'"
                )

    @pytest.mark.parametrize("task_name", SEARCH_TASKS)
    def test_relevance_items_exist_in_corpus(self, task_name):
        """Every item referenced in a query's relevance dict must be in the corpus."""
        data = _load_data_file(task_name)
        corpus_set = set(data["corpus"])
        for q in data["queries"]:
            for item in q["relevance"]:
                assert item in corpus_set, (
                    f"{task_name}: query {q['id']} references '{item}' "
                    f"which is not in the corpus"
                )

    @pytest.mark.parametrize("task_name", SEARCH_TASKS)
    def test_no_duplicate_corpus_items(self, task_name):
        data = _load_data_file(task_name)
        corpus = data["corpus"]
        assert len(corpus) == len(set(corpus)), (
            f"{task_name}: duplicate corpus items found"
        )

    @pytest.mark.parametrize("task_name", SEARCH_TASKS)
    def test_corpus_items_are_nonempty_strings(self, task_name):
        data = _load_data_file(task_name)
        for item in data["corpus"]:
            assert isinstance(item, str) and len(item.strip()) > 0, (
                f"{task_name}: corpus contains empty or non-string item"
            )

    @pytest.mark.parametrize("task_name", SEARCH_TASKS)
    def test_every_query_has_at_least_one_relevant(self, task_name):
        """A query with zero relevant items is useless for NDCG evaluation."""
        data = _load_data_file(task_name)
        for q in data["queries"]:
            has_relevant = any(g > 0 for g in q["relevance"].values())
            assert has_relevant, (
                f"{task_name}: query {q['id']} has no relevant items (all grades are 0)"
            )

    @pytest.mark.parametrize("task_name", SEARCH_TASKS)
    def test_metadata_query_count_matches(self, task_name):
        data = _load_data_file(task_name)
        if "metadata" in data and "n_queries" in data["metadata"]:
            assert data["metadata"]["n_queries"] == len(data["queries"]), (
                f"{task_name}: metadata.n_queries disagrees with actual query count"
            )

    @pytest.mark.parametrize("task_name", SEARCH_TASKS)
    def test_metadata_corpus_count_matches(self, task_name):
        data = _load_data_file(task_name)
        if "metadata" in data and "n_corpus" in data["metadata"]:
            assert data["metadata"]["n_corpus"] == len(data["corpus"]), (
                f"{task_name}: metadata.n_corpus disagrees with actual corpus count"
            )

    @pytest.mark.parametrize("task_name", SEARCH_TASKS)
    def test_query_texts_are_nonempty(self, task_name):
        """Query strings should be non-empty."""
        data = _load_data_file(task_name)
        for q in data["queries"]:
            assert isinstance(q["query"], str) and len(q["query"].strip()) > 0, (
                f"{task_name}: query {q['id']} has empty query text"
            )

    @pytest.mark.parametrize("task_name", SEARCH_TASKS)
    def test_queries_have_domain_field(self, task_name):
        """Every query should have a domain annotation."""
        data = _load_data_file(task_name)
        for q in data["queries"]:
            assert "domain" in q, f"{task_name}: query {q['id']} missing 'domain' field"


# =========================================================================
# Pair classification data integrity
# =========================================================================


class TestPairTaskData:
    """Structure and content validation for pair classification tasks."""

    @pytest.mark.parametrize("task_name", PAIR_TASKS)
    def test_required_top_level_keys(self, task_name):
        data = _load_data_file(task_name)
        for key in ("task", "version", "description", "metric", "pairs"):
            assert key in data, f"{task_name}: missing top-level key '{key}'"

    @pytest.mark.parametrize("task_name", PAIR_TASKS)
    def test_minimum_pair_count(self, task_name):
        """Each pair task should have at least 100 pairs."""
        data = _load_data_file(task_name)
        assert len(data["pairs"]) >= 100, (
            f"{task_name}: only {len(data['pairs'])} pairs (minimum 100)"
        )

    @pytest.mark.parametrize("task_name", PAIR_TASKS)
    def test_pairs_have_required_fields(self, task_name):
        data = _load_data_file(task_name)
        for p in data["pairs"]:
            assert "id" in p, f"{task_name}: pair missing 'id'"
            assert "text_a" in p, f"{task_name}: pair {p['id']} missing 'text_a'"
            assert "text_b" in p, f"{task_name}: pair {p['id']} missing 'text_b'"
            assert "label" in p, f"{task_name}: pair {p['id']} missing 'label'"

    @pytest.mark.parametrize("task_name", PAIR_TASKS)
    def test_labels_are_binary(self, task_name):
        data = _load_data_file(task_name)
        for p in data["pairs"]:
            assert p["label"] in (0, 1), (
                f"{task_name}: pair {p['id']} has non-binary label {p['label']}"
            )

    @pytest.mark.parametrize("task_name", PAIR_TASKS)
    def test_no_duplicate_pair_ids(self, task_name):
        data = _load_data_file(task_name)
        ids = [p["id"] for p in data["pairs"]]
        assert len(ids) == len(set(ids)), f"{task_name}: duplicate pair IDs found"

    @pytest.mark.parametrize("task_name", PAIR_TASKS)
    def test_has_both_positive_and_negative_labels(self, task_name):
        """A pair classification benchmark needs both classes to be meaningful."""
        data = _load_data_file(task_name)
        labels = {p["label"] for p in data["pairs"]}
        assert 0 in labels, f"{task_name}: no negative (0) labels found"
        assert 1 in labels, f"{task_name}: no positive (1) labels found"

    @pytest.mark.parametrize("task_name", PAIR_TASKS)
    def test_texts_are_nonempty(self, task_name):
        data = _load_data_file(task_name)
        for p in data["pairs"]:
            assert isinstance(p["text_a"], str) and len(p["text_a"].strip()) > 0, (
                f"{task_name}: pair {p['id']} has empty text_a"
            )
            assert isinstance(p["text_b"], str) and len(p["text_b"].strip()) > 0, (
                f"{task_name}: pair {p['id']} has empty text_b"
            )

    @pytest.mark.parametrize("task_name", PAIR_TASKS)
    def test_metadata_pair_count_matches(self, task_name):
        data = _load_data_file(task_name)
        if "metadata" in data and "n_pairs" in data["metadata"]:
            assert data["metadata"]["n_pairs"] == len(data["pairs"]), (
                f"{task_name}: metadata.n_pairs disagrees with actual pair count"
            )

    @pytest.mark.parametrize("task_name", PAIR_TASKS)
    def test_metadata_positive_negative_counts(self, task_name):
        data = _load_data_file(task_name)
        if "metadata" not in data:
            return
        meta = data["metadata"]
        actual_pos = sum(1 for p in data["pairs"] if p["label"] == 1)
        actual_neg = sum(1 for p in data["pairs"] if p["label"] == 0)
        if "n_positive" in meta:
            assert meta["n_positive"] == actual_pos, (
                f"{task_name}: metadata.n_positive={meta['n_positive']} "
                f"but actual={actual_pos}"
            )
        if "n_negative" in meta:
            assert meta["n_negative"] == actual_neg, (
                f"{task_name}: metadata.n_negative={meta['n_negative']} "
                f"but actual={actual_neg}"
            )

    @pytest.mark.parametrize("task_name", PAIR_TASKS)
    def test_pairs_have_domain_field(self, task_name):
        """Multi-domain pair tasks should have a domain annotation on each pair.

        Single-domain tasks (split match files, portion_size, noisy_menu_match)
        are already domain-specific, so per-pair domain is optional.
        """
        data = _load_data_file(task_name)
        single_domain_tasks = {
            "indian_match",
            "global_match",
            "beverage_match",
            "bakery_match",
            "portion_size",
            "noisy_menu_match",
        }
        if task_name in single_domain_tasks:
            return
        for p in data["pairs"]:
            assert "domain" in p, f"{task_name}: pair {p['id']} missing 'domain' field"

    @pytest.mark.parametrize("task_name", PAIR_TASKS)
    def test_label_balance_not_extreme(self, task_name):
        """Neither class should have less than 20% of all pairs."""
        data = _load_data_file(task_name)
        n_total = len(data["pairs"])
        n_pos = sum(1 for p in data["pairs"] if p["label"] == 1)
        ratio = n_pos / n_total
        assert 0.2 <= ratio <= 0.8, (
            f"{task_name}: extreme label imbalance "
            f"({n_pos}/{n_total} = {ratio:.2f} positive)"
        )


# =========================================================================
# Classification data integrity
# =========================================================================


class TestClassificationTaskData:
    """Structure and content validation for the classification task."""

    @pytest.mark.parametrize("task_name", CLASSIFICATION_TASKS)
    def test_required_top_level_keys(self, task_name):
        data = _load_data_file(task_name)
        for key in ("task", "version", "description", "metric", "items", "label_names"):
            assert key in data, f"{task_name}: missing top-level key '{key}'"

    @pytest.mark.parametrize("task_name", CLASSIFICATION_TASKS)
    def test_minimum_item_count(self, task_name):
        """Classification should have at least 200 items."""
        data = _load_data_file(task_name)
        assert len(data["items"]) >= 200, (
            f"{task_name}: only {len(data['items'])} items (minimum 200)"
        )

    @pytest.mark.parametrize("task_name", CLASSIFICATION_TASKS)
    def test_items_have_required_fields(self, task_name):
        data = _load_data_file(task_name)
        for item in data["items"]:
            assert "id" in item, f"{task_name}: item missing 'id'"
            assert "text" in item, (
                f"{task_name}: item {item.get('id', '?')} missing 'text'"
            )
            assert "label" in item, (
                f"{task_name}: item {item.get('id', '?')} missing 'label'"
            )

    @pytest.mark.parametrize("task_name", CLASSIFICATION_TASKS)
    def test_no_duplicate_item_ids(self, task_name):
        data = _load_data_file(task_name)
        ids = [item["id"] for item in data["items"]]
        assert len(ids) == len(set(ids)), f"{task_name}: duplicate item IDs found"

    @pytest.mark.parametrize("task_name", CLASSIFICATION_TASKS)
    def test_all_labels_in_label_names(self, task_name):
        """Every label appearing in items must be listed in label_names."""
        data = _load_data_file(task_name)
        label_names_set = set(data["label_names"])
        for item in data["items"]:
            assert item["label"] in label_names_set, (
                f"{task_name}: item {item['id']} has label '{item['label']}' "
                f"not in label_names"
            )

    @pytest.mark.parametrize("task_name", CLASSIFICATION_TASKS)
    def test_label_names_all_used(self, task_name):
        """Every entry in label_names should appear in at least one item."""
        data = _load_data_file(task_name)
        used_labels = {item["label"] for item in data["items"]}
        for name in data["label_names"]:
            assert name in used_labels, (
                f"{task_name}: label_names contains '{name}' but no item has that label"
            )

    @pytest.mark.parametrize("task_name", CLASSIFICATION_TASKS)
    def test_texts_are_nonempty(self, task_name):
        data = _load_data_file(task_name)
        for item in data["items"]:
            assert isinstance(item["text"], str) and len(item["text"].strip()) > 0, (
                f"{task_name}: item {item['id']} has empty text"
            )

    @pytest.mark.parametrize("task_name", CLASSIFICATION_TASKS)
    def test_metadata_item_count_matches(self, task_name):
        data = _load_data_file(task_name)
        if "metadata" in data and "n_items" in data["metadata"]:
            assert data["metadata"]["n_items"] == len(data["items"]), (
                f"{task_name}: metadata.n_items disagrees with actual item count"
            )

    @pytest.mark.parametrize("task_name", CLASSIFICATION_TASKS)
    def test_metadata_class_count_matches(self, task_name):
        data = _load_data_file(task_name)
        if "metadata" in data and "n_classes" in data["metadata"]:
            assert data["metadata"]["n_classes"] == len(data["label_names"]), (
                f"{task_name}: metadata.n_classes disagrees with label_names length"
            )

    @pytest.mark.parametrize("task_name", CLASSIFICATION_TASKS)
    def test_minimum_class_count(self, task_name):
        """Need at least 3 classes for a meaningful classification benchmark."""
        data = _load_data_file(task_name)
        assert len(data["label_names"]) >= 3, (
            f"{task_name}: only {len(data['label_names'])} classes (minimum 3)"
        )

    @pytest.mark.parametrize("task_name", CLASSIFICATION_TASKS)
    def test_each_class_has_minimum_examples(self, task_name):
        """Every class should have at least 3 examples for stratified splitting."""
        data = _load_data_file(task_name)
        label_counts = Counter(item["label"] for item in data["items"])
        for label, count in label_counts.items():
            assert count >= 3, (
                f"{task_name}: class '{label}' has only {count} examples (minimum 3)"
            )

    @pytest.mark.parametrize("task_name", CLASSIFICATION_TASKS)
    def test_no_duplicate_texts(self, task_name):
        """Duplicate texts within the classification data create data leakage."""
        data = _load_data_file(task_name)
        texts = [item["text"] for item in data["items"]]
        duplicates = [t for t, c in Counter(texts).items() if c > 1]
        assert len(duplicates) == 0, (
            f"{task_name}: {len(duplicates)} duplicate texts found: {duplicates[:3]}"
        )

    @pytest.mark.parametrize("task_name", CLASSIFICATION_TASKS)
    def test_label_names_are_sorted(self, task_name):
        """label_names should be sorted for deterministic label-to-index mapping."""
        data = _load_data_file(task_name)
        assert data["label_names"] == sorted(data["label_names"]), (
            f"{task_name}: label_names is not sorted"
        )
