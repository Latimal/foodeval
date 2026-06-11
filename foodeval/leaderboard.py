"""Leaderboard generation from FoodEval result JSON files.

Reads saved result files and renders them as category-grouped markdown
leaderboard tables. Models are ranked by FoodEval Score, the unweighted
mean across all 12 task scores (1/12 each). Category averages (Search,
Matching, Classification) are shown for capability-level comparison.

Usage:
    >>> from foodeval.leaderboard import generate_leaderboard
    >>> markdown = generate_leaderboard("results/")  # doctest: +SKIP
    >>> print(markdown)  # doctest: +SKIP

    $ python -m foodeval leaderboard results/
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from foodeval.tasks import get_task

# ---------------------------------------------------------------------------
# Task categories and display metadata
# ---------------------------------------------------------------------------

TASK_CATEGORIES = {
    "Search": ["food_search", "concept_search", "diet_search", "noisy_search"],
    "Matching": [
        "indian_match",
        "global_match",
        "beverage_match",
        "bakery_match",
        "portion_size",
        "noisy_menu_match",
        "cross_lingual_match",
    ],
    "Classification": ["cuisine_classify"],
}

CATEGORY_ORDER = ["Search", "Matching", "Classification"]

CATEGORY_METRICS = {
    "Search": "NDCG@10",
    "Matching": "Best F1",
    "Classification": "Macro F1",
}

TASK_ORDER = [
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
]

TASK_SHORT_NAMES = {
    "food_search": "Food",
    "concept_search": "Concept",
    "diet_search": "Diet",
    "noisy_search": "Noisy",
    "indian_match": "Indian",
    "global_match": "Global",
    "beverage_match": "Bev",
    "bakery_match": "Bakery",
    "portion_size": "Portion",
    "noisy_menu_match": "Noisy Menu",
    "cross_lingual_match": "X-Lingual",
    "cuisine_classify": "Cuisine",
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def compute_category_scores(task_scores: dict[str, float]) -> dict[str, float]:
    """Compute the mean score for each category from per-task scores."""
    cat_scores = {}
    for cat, tasks in TASK_CATEGORIES.items():
        values = [
            task_scores[t] for t in tasks if t in task_scores and task_scores[t] > 0
        ]
        cat_scores[cat] = sum(values) / len(values) if values else 0.0
    return cat_scores


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class LeaderboardEntry:
    """A single model's scores for leaderboard display."""

    model_name: str
    dimension: int
    task_scores: dict[str, float]
    timestamp: str
    source_file: str
    category_scores: dict[str, float] = field(default_factory=dict)
    foodeval_score: float = 0.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_result_file(path: Path) -> LeaderboardEntry | None:
    """Parse a single result JSON into a LeaderboardEntry."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"Warning: skipping unreadable result file {path}: {exc}",
            file=sys.stderr,
        )
        return None

    tasks = data.get("tasks", {})
    task_scores = {}
    for task_name, task_data in tasks.items():
        if isinstance(task_data, dict):
            task_scores[task_name] = task_data.get("main_score", 0.0)

    cat_scores = compute_category_scores(task_scores)
    # FoodEval Score = task-equal average (1/N across all tasks).
    # This is the standard approach (MTEB, BEIR, SuperGLUE). Each task
    # contributes equally regardless of which category it belongs to.
    agg = data.get("aggregate_score", 0.0)
    if not agg and task_scores:
        agg = round(sum(task_scores.values()) / len(task_scores), 4)

    return LeaderboardEntry(
        model_name=data.get("model_name", path.stem),
        dimension=data.get("dimension", 0),
        task_scores=task_scores,
        timestamp=data.get("timestamp", ""),
        source_file=str(path),
        category_scores=cat_scores,
        foodeval_score=agg,
    )


def load_results(results_dir: str) -> list[LeaderboardEntry]:
    """Load all result JSON files from a directory.

    Returns list sorted by FoodEval Score descending.
    """
    results_path = Path(results_dir)
    if not results_path.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    entries: list[LeaderboardEntry] = []
    for json_file in sorted(results_path.glob("*.json")):
        entry = _load_result_file(json_file)
        if entry is not None:
            entries.append(entry)

    entries.sort(key=lambda e: (-e.foodeval_score, e.model_name))
    return entries


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def format_overall_table(entries: list[LeaderboardEntry]) -> str:
    """Overall leaderboard showing category averages and FoodEval Score."""
    if not entries:
        return "No results found."

    header = "| Rank | Model | Dim | Search | Matching | Classification | **FoodEval Score** |"
    sep = "|------|-------|----:|-------:|---------:|---------------:|-------------------:|"
    rows = [header, sep]

    for rank, e in enumerate(entries, 1):
        s = e.category_scores.get("Search", 0.0)
        m = e.category_scores.get("Matching", 0.0)
        c = e.category_scores.get("Classification", 0.0)
        rows.append(
            f"| {rank} | {e.model_name} | {e.dimension} "
            f"| {s:.4f} | {m:.4f} | {c:.4f} "
            f"| **{e.foodeval_score:.4f}** |"
        )
    return "\n".join(rows)


def format_category_table(entries: list[LeaderboardEntry], category: str) -> str:
    """Sub-leaderboard for a single category, ranked by that category's mean."""
    if not entries:
        return "No results found."

    tasks = TASK_CATEGORIES[category]
    active_tasks = [
        t for t in tasks if any(e.task_scores.get(t, 0.0) > 0.0 for e in entries)
    ]
    if not active_tasks:
        return "No results found."

    sorted_entries = sorted(
        entries, key=lambda e: (-e.category_scores.get(category, 0.0), e.model_name)
    )

    task_headers = [TASK_SHORT_NAMES.get(t, t) for t in active_tasks]
    header = "| Rank | Model | " + " | ".join(task_headers) + " | **Avg** |"
    sep = "|------|-------" + "|------:" * len(active_tasks) + "|--------:|"
    rows = [header, sep]

    for rank, e in enumerate(sorted_entries, 1):
        scores = []
        for t in active_tasks:
            s = e.task_scores.get(t, 0.0)
            scores.append(f"{s:.4f}" if s > 0 else "-")
        cat_avg = e.category_scores.get(category, 0.0)
        rows.append(
            f"| {rank} | {e.model_name} | "
            + " | ".join(scores)
            + f" | **{cat_avg:.4f}** |"
        )
    return "\n".join(rows)


def format_result_table(entries: list[LeaderboardEntry]) -> str:
    """Full flat table with all 12 tasks (backward-compatible)."""
    if not entries:
        return "No results found."

    active_tasks = [
        t for t in TASK_ORDER if any(e.task_scores.get(t, 0.0) > 0.0 for e in entries)
    ]
    task_headers = [TASK_SHORT_NAMES.get(t, t) for t in active_tasks]
    header = "| Rank | Model | Dim | " + " | ".join(task_headers) + " | **FoodEval** |"
    separator = "|------|-------|----:" + "|------:" * len(active_tasks) + "|--------:|"

    rows: list[str] = [header, separator]

    for rank, entry in enumerate(entries, 1):
        scores = []
        for t in active_tasks:
            s = entry.task_scores.get(t, 0.0)
            scores.append(f"{s:.4f}" if s > 0 else "-")

        row = (
            f"| {rank} | {entry.model_name} | {entry.dimension} | "
            + " | ".join(scores)
            + f" | **{entry.foodeval_score:.4f}** |"
        )
        rows.append(row)

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Full leaderboard generation
# ---------------------------------------------------------------------------


def generate_leaderboard(results_dir: str) -> str:
    """Generate the full category-grouped markdown leaderboard."""
    entries = load_results(results_dir)

    lines: list[str] = []
    lines.append("# FoodEval Leaderboard")
    lines.append("")
    lines.append(
        f"_{len(entries)} model{'s' if len(entries) != 1 else ''} evaluated. "
        f"FoodEval Score = unweighted mean across all 12 tasks (1/12 each). "
        f"Category averages shown for capability-specific comparison._"
    )
    lines.append("")

    # Overall table
    lines.append("## Overall")
    lines.append("")
    lines.append(format_overall_table(entries))
    lines.append("")

    # Per-category sub-leaderboards
    for cat in CATEGORY_ORDER:
        tasks = TASK_CATEGORIES[cat]
        metric = CATEGORY_METRICS[cat]
        lines.append(f"## {cat}")
        lines.append(f"_{metric}, {len(tasks)} task{'s' if len(tasks) != 1 else ''}._")
        lines.append("")
        lines.append(format_category_table(entries, cat))
        lines.append("")

    # Legend
    lines.append("**Tasks:**")
    for cat in CATEGORY_ORDER:
        lines.append(f"- *{cat}*")
        for t in TASK_CATEGORIES[cat]:
            short = TASK_SHORT_NAMES.get(t, t)
            try:
                task_obj = get_task(t)
                metric_label = task_obj.metric_name
            except KeyError:
                metric_label = t
            lines.append(f"  - {short} ({t}): {metric_label}")
    lines.append("")

    return "\n".join(lines)
