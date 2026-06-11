"""Tests for the leaderboard module (foodeval.leaderboard).

The leaderboard loads result JSON files, ranks models by FoodEval Score
(unweighted mean across all tasks), and renders category-grouped
markdown tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foodeval.leaderboard import (
    CATEGORY_ORDER,
    TASK_CATEGORIES,
    TASK_ORDER,
    TASK_SHORT_NAMES,
    LeaderboardEntry,
    compute_category_scores,
    format_category_table,
    format_overall_table,
    format_result_table,
    generate_leaderboard,
    load_results,
)


# =========================================================================
# Helpers
# =========================================================================


def _write_result(path: Path, model_name: str, aggregate: float, tasks: dict) -> Path:
    """Write a minimal result JSON file."""
    data = {
        "model_name": model_name,
        "dimension": 384,
        "aggregate_score": aggregate,
        "timestamp": "2026-01-01T00:00:00Z",
        "total_seconds": 10.0,
        "tasks": {
            name: {"task_name": name, "main_score": score, "metric_name": "ndcg@10"}
            for name, score in tasks.items()
        },
    }
    result_file = path / f"{model_name}.json"
    result_file.write_text(json.dumps(data), encoding="utf-8")
    return result_file


# Full task set across all categories for tests that need coverage.
ALL_TASKS = {
    "food_search": 0.80,
    "concept_search": 0.70,
    "diet_search": 0.75,
    "noisy_search": 0.65,
    "indian_match": 0.90,
    "global_match": 0.85,
    "beverage_match": 0.60,
    "bakery_match": 0.55,
    "portion_size": 0.70,
    "noisy_menu_match": 0.65,
    "cross_lingual_match": 0.50,
    "cuisine_classify": 0.78,
}


def _make_entry(
    name: str = "test-model",
    dim: int = 384,
    task_scores: dict | None = None,
) -> LeaderboardEntry:
    """Build a LeaderboardEntry with computed category/foodeval scores."""
    ts = task_scores or {"food_search": 0.80}
    cat = compute_category_scores(ts)
    fe = round(sum(ts.values()) / len(ts), 4)
    return LeaderboardEntry(
        model_name=name,
        dimension=dim,
        task_scores=ts,
        timestamp="2026-01-01",
        source_file="test.json",
        category_scores=cat,
        foodeval_score=fe,
    )


# =========================================================================
# compute_category_scores
# =========================================================================


class TestComputeCategoryScores:
    """Compute per-category mean from per-task scores."""

    def test_full_coverage(self):
        cat = compute_category_scores(ALL_TASKS)
        # Search: mean(0.80, 0.70, 0.75, 0.65) = 0.725
        assert abs(cat["Search"] - 0.725) < 1e-9
        # Matching: mean(0.90, 0.85, 0.60, 0.55, 0.70, 0.65, 0.50) = 0.6785714...
        assert (
            abs(cat["Matching"] - (0.90 + 0.85 + 0.60 + 0.55 + 0.70 + 0.65 + 0.50) / 7)
            < 1e-9
        )
        # Classification: 0.78
        assert abs(cat["Classification"] - 0.78) < 1e-9

    def test_partial_tasks(self):
        """Only some tasks present; missing ones are excluded from the mean."""
        cat = compute_category_scores({"food_search": 0.90, "indian_match": 0.70})
        assert abs(cat["Search"] - 0.90) < 1e-9
        assert abs(cat["Matching"] - 0.70) < 1e-9
        assert cat["Classification"] == 0.0

    def test_zero_scores_excluded(self):
        """Tasks with score 0.0 are excluded from the category mean."""
        cat = compute_category_scores({"food_search": 0.80, "concept_search": 0.0})
        # Only food_search counted
        assert abs(cat["Search"] - 0.80) < 1e-9

    def test_empty_input(self):
        cat = compute_category_scores({})
        assert cat["Search"] == 0.0
        assert cat["Matching"] == 0.0
        assert cat["Classification"] == 0.0

    def test_unknown_tasks_ignored(self):
        """Tasks not in any category are silently ignored."""
        cat = compute_category_scores({"unknown_task": 0.99, "food_search": 0.70})
        assert abs(cat["Search"] - 0.70) < 1e-9
        assert cat["Matching"] == 0.0


# =========================================================================
# load_results
# =========================================================================


class TestLoadResults:
    """Load and parse result JSON files from a directory."""

    def test_loads_single_file(self, tmp_path):
        _write_result(tmp_path, "model-a", 0.85, {"food_search": 0.85})
        entries = load_results(str(tmp_path))
        assert len(entries) == 1
        assert entries[0].model_name == "model-a"

    def test_loads_multiple_files(self, tmp_path):
        _write_result(tmp_path, "model-a", 0.85, {"food_search": 0.85})
        _write_result(tmp_path, "model-b", 0.75, {"food_search": 0.75})
        entries = load_results(str(tmp_path))
        assert len(entries) == 2

    def test_sorts_by_foodeval_score_descending(self, tmp_path):
        _write_result(tmp_path, "low", 0.60, {"food_search": 0.60})
        _write_result(tmp_path, "high", 0.90, {"food_search": 0.90})
        _write_result(tmp_path, "mid", 0.75, {"food_search": 0.75})
        entries = load_results(str(tmp_path))
        scores = [e.foodeval_score for e in entries]
        assert scores == sorted(scores, reverse=True)

    def test_nonexistent_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            load_results("/tmp/foodeval_nonexistent_xyz")

    def test_empty_dir_returns_empty_list(self, tmp_path):
        entries = load_results(str(tmp_path))
        assert entries == []

    def test_skips_invalid_json_files(self, tmp_path):
        _write_result(tmp_path, "valid-model", 0.80, {"food_search": 0.80})
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("this is not json", encoding="utf-8")
        entries = load_results(str(tmp_path))
        assert len(entries) == 1

    def test_ignores_non_json_files(self, tmp_path):
        _write_result(tmp_path, "valid-model", 0.80, {"food_search": 0.80})
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("just a note", encoding="utf-8")
        entries = load_results(str(tmp_path))
        assert len(entries) == 1

    def test_entry_has_correct_fields(self, tmp_path):
        _write_result(
            tmp_path,
            "test-model",
            0.82,
            {"food_search": 0.85, "indian_match": 0.79},
        )
        entries = load_results(str(tmp_path))
        entry = entries[0]
        assert entry.model_name == "test-model"
        assert entry.dimension == 384
        assert entry.foodeval_score == 0.82
        assert entry.task_scores["food_search"] == 0.85
        assert entry.task_scores["indian_match"] == 0.79

    def test_entry_has_category_scores(self, tmp_path):
        _write_result(tmp_path, "m", 0.80, {"food_search": 0.80, "indian_match": 0.70})
        entry = load_results(str(tmp_path))[0]
        assert abs(entry.category_scores["Search"] - 0.80) < 1e-9
        assert abs(entry.category_scores["Matching"] - 0.70) < 1e-9
        assert entry.category_scores["Classification"] == 0.0

    def test_entry_has_foodeval_score(self, tmp_path):
        """foodeval_score is read from the file's aggregate_score field."""
        _write_result(tmp_path, "m", 0.80, {"food_search": 0.80, "indian_match": 0.70})
        entry = load_results(str(tmp_path))[0]
        assert abs(entry.foodeval_score - 0.80) < 1e-9

    def test_foodeval_score_fallback_is_task_mean(self, tmp_path):
        """Without an aggregate_score field, the unweighted task mean is used."""
        data = {
            "model_name": "no-agg",
            "dimension": 384,
            "tasks": {
                "food_search": {"task_name": "food_search", "main_score": 0.80},
                "indian_match": {"task_name": "indian_match", "main_score": 0.70},
            },
        }
        (tmp_path / "no-agg.json").write_text(json.dumps(data), encoding="utf-8")
        entry = load_results(str(tmp_path))[0]
        assert abs(entry.foodeval_score - 0.75) < 1e-9

    def test_tiebreaker_is_model_name(self, tmp_path):
        """Models with the same foodeval_score should be sorted by name."""
        _write_result(tmp_path, "zebra", 0.80, {"food_search": 0.80})
        _write_result(tmp_path, "alpha", 0.80, {"food_search": 0.80})
        entries = load_results(str(tmp_path))
        assert entries[0].model_name == "alpha"
        assert entries[1].model_name == "zebra"


# =========================================================================
# format_result_table
# =========================================================================


class TestFormatResultTable:
    """Render leaderboard entries as a flat markdown table."""

    def test_empty_entries_returns_message(self):
        result = format_result_table([])
        assert "No results" in result

    def test_single_entry_produces_table(self):
        entry = _make_entry("test-model", task_scores={"food_search": 0.85})
        table = format_result_table([entry])
        assert "test-model" in table
        assert "384" in table
        assert "0.85" in table

    def test_table_has_header_and_separator(self):
        entry = _make_entry("test", dim=64, task_scores={"food_search": 0.80})
        table = format_result_table([entry])
        lines = table.split("\n")
        assert "|" in lines[0]  # header
        assert "---" in lines[1]  # separator

    def test_table_shows_rank(self):
        entries = [
            _make_entry("a", task_scores={"food_search": 0.90}),
            _make_entry("b", task_scores={"food_search": 0.80}),
        ]
        table = format_result_table(entries)
        assert "| 1 |" in table
        assert "| 2 |" in table

    def test_only_active_tasks_shown(self):
        """Tasks where all models scored 0 should be hidden."""
        entry = _make_entry(
            task_scores={"food_search": 0.80, "indian_match": 0.0},
        )
        table = format_result_table([entry])
        assert "Food" in table
        # indian_match has 0.0 score so its short name should not appear as a column
        assert "Indian" not in table

    def test_last_column_is_foodeval(self):
        entry = _make_entry(task_scores={"food_search": 0.80})
        table = format_result_table([entry])
        header = table.split("\n")[0]
        assert "FoodEval" in header


# =========================================================================
# format_overall_table
# =========================================================================


class TestFormatOverallTable:
    """Overall leaderboard with category columns and FoodEval Score."""

    def test_empty_entries_returns_message(self):
        result = format_overall_table([])
        assert "No results" in result

    def test_has_category_columns(self):
        entry = _make_entry(task_scores=ALL_TASKS)
        table = format_overall_table([entry])
        header = table.split("\n")[0]
        assert "Search" in header
        assert "Matching" in header
        assert "Classification" in header

    def test_has_foodeval_score_column(self):
        entry = _make_entry(task_scores=ALL_TASKS)
        table = format_overall_table([entry])
        header = table.split("\n")[0]
        assert "FoodEval Score" in header

    def test_shows_rank_and_model(self):
        entries = [
            _make_entry("top-model", task_scores=ALL_TASKS),
            _make_entry("other", task_scores={"food_search": 0.50}),
        ]
        table = format_overall_table(entries)
        assert "| 1 |" in table
        assert "| 2 |" in table
        assert "top-model" in table

    def test_shows_category_scores(self):
        entry = _make_entry(task_scores={"food_search": 0.80, "cuisine_classify": 0.60})
        table = format_overall_table([entry])
        assert "0.8000" in table
        assert "0.6000" in table


# =========================================================================
# format_category_table
# =========================================================================


class TestFormatCategoryTable:
    """Per-category sub-leaderboard showing only that category's tasks."""

    def test_empty_entries_returns_message(self):
        result = format_category_table([], "Search")
        assert "No results" in result

    def test_search_table_shows_only_search_tasks(self):
        entry = _make_entry(task_scores=ALL_TASKS)
        table = format_category_table([entry], "Search")
        # Search short names present
        assert "Food" in table
        assert "Concept" in table
        assert "Diet" in table
        assert "Noisy" in table
        # Matching short names absent
        assert "Indian" not in table
        assert "Global" not in table

    def test_matching_table_shows_only_matching_tasks(self):
        entry = _make_entry(task_scores=ALL_TASKS)
        table = format_category_table([entry], "Matching")
        assert "Indian" in table
        assert "Global" in table
        # Search short names absent
        assert "Concept" not in table

    def test_classification_table(self):
        entry = _make_entry(task_scores={"cuisine_classify": 0.78})
        table = format_category_table([entry], "Classification")
        assert "Cuisine" in table

    def test_ranked_by_category_score(self):
        e1 = _make_entry(
            "low", task_scores={"food_search": 0.50, "concept_search": 0.50}
        )
        e2 = _make_entry(
            "high", task_scores={"food_search": 0.90, "concept_search": 0.90}
        )
        table = format_category_table([e1, e2], "Search")
        lines = [ln for ln in table.split("\n") if "high" in ln or "low" in ln]
        # high should appear first (rank 1)
        assert lines[0].index("high") < lines[1].index("low") or "| 1 |" in lines[0]

    def test_has_avg_column(self):
        entry = _make_entry(task_scores=ALL_TASKS)
        table = format_category_table([entry], "Search")
        header = table.split("\n")[0]
        assert "Avg" in header

    def test_inactive_tasks_hidden(self):
        """If no model has a score for a task, it should be hidden."""
        entry = _make_entry(task_scores={"food_search": 0.80})
        table = format_category_table([entry], "Search")
        assert "Food" in table
        # concept_search, diet_search, noisy_search all 0 or missing
        assert "Concept" not in table

    def test_no_active_tasks_returns_message(self):
        """Category with zero scores for all its tasks."""
        entry = _make_entry(task_scores={"food_search": 0.80})
        # Classification has no scores
        table = format_category_table([entry], "Classification")
        assert "No results" in table


# =========================================================================
# generate_leaderboard
# =========================================================================


class TestGenerateLeaderboard:
    """Full leaderboard generation from result directory."""

    def test_generates_markdown_with_title(self, tmp_path):
        _write_result(tmp_path, "model-a", 0.85, {"food_search": 0.85})
        md = generate_leaderboard(str(tmp_path))
        assert "# FoodEval Leaderboard" in md

    def test_includes_model_count(self, tmp_path):
        _write_result(tmp_path, "model-a", 0.85, {"food_search": 0.85})
        _write_result(tmp_path, "model-b", 0.75, {"food_search": 0.75})
        md = generate_leaderboard(str(tmp_path))
        assert "2 models" in md

    def test_singular_model_count(self, tmp_path):
        _write_result(tmp_path, "model-a", 0.85, {"food_search": 0.85})
        md = generate_leaderboard(str(tmp_path))
        assert "1 model " in md

    def test_includes_task_legend(self, tmp_path):
        _write_result(tmp_path, "model-a", 0.85, {"food_search": 0.85})
        md = generate_leaderboard(str(tmp_path))
        assert "Tasks:" in md

    def test_nonexistent_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            generate_leaderboard("/tmp/foodeval_nonexistent_xyz")

    def test_empty_dir_still_produces_output(self, tmp_path):
        md = generate_leaderboard(str(tmp_path))
        assert "Leaderboard" in md
        assert "0 models" in md

    def test_includes_overall_section(self, tmp_path):
        _write_result(tmp_path, "model-a", 0.85, ALL_TASKS)
        md = generate_leaderboard(str(tmp_path))
        assert "## Overall" in md

    def test_includes_category_sub_leaderboards(self, tmp_path):
        _write_result(tmp_path, "model-a", 0.85, ALL_TASKS)
        md = generate_leaderboard(str(tmp_path))
        assert "## Search" in md
        assert "## Matching" in md
        assert "## Classification" in md

    def test_includes_foodeval_score_description(self, tmp_path):
        _write_result(tmp_path, "model-a", 0.85, {"food_search": 0.85})
        md = generate_leaderboard(str(tmp_path))
        assert "FoodEval Score" in md

    def test_legend_groups_by_category(self, tmp_path):
        _write_result(tmp_path, "model-a", 0.85, ALL_TASKS)
        md = generate_leaderboard(str(tmp_path))
        # Legend should list category names
        for cat in CATEGORY_ORDER:
            assert f"*{cat}*" in md


# =========================================================================
# Constants
# =========================================================================


class TestLeaderboardConstants:
    """Verify leaderboard configuration is consistent."""

    def test_task_order_has_all_known_tasks(self):
        expected = {
            "food_search",
            "concept_search",
            "diet_search",
            "noisy_search",
            "indian_match",
            "global_match",
            "beverage_match",
            "bakery_match",
            "portion_size",
            "noisy_menu_match",
            "cross_lingual_match",
            "cuisine_classify",
        }
        assert set(TASK_ORDER) == expected

    def test_task_short_names_covers_task_order(self):
        for task in TASK_ORDER:
            assert task in TASK_SHORT_NAMES, (
                f"TASK_SHORT_NAMES missing entry for '{task}'"
            )

    def test_task_categories_covers_task_order(self):
        """Every task in TASK_ORDER must appear in exactly one category."""
        categorized = set()
        for tasks in TASK_CATEGORIES.values():
            for t in tasks:
                assert t not in categorized, (
                    f"Task '{t}' appears in multiple categories"
                )
                categorized.add(t)
        assert categorized == set(TASK_ORDER)

    def test_category_order_matches_task_categories_keys(self):
        assert set(CATEGORY_ORDER) == set(TASK_CATEGORIES.keys())

    def test_category_order_length(self):
        assert len(CATEGORY_ORDER) == 3

    def test_search_category_tasks(self):
        assert TASK_CATEGORIES["Search"] == [
            "food_search",
            "concept_search",
            "diet_search",
            "noisy_search",
        ]

    def test_matching_category_tasks(self):
        assert TASK_CATEGORIES["Matching"] == [
            "indian_match",
            "global_match",
            "beverage_match",
            "bakery_match",
            "portion_size",
            "noisy_menu_match",
            "cross_lingual_match",
        ]

    def test_classification_category_tasks(self):
        assert TASK_CATEGORIES["Classification"] == ["cuisine_classify"]
