"""FoodEval task registry.

Provides a central registry of all benchmark tasks, with helpers to look up
tasks by name or retrieve the full set.

Usage:
    >>> from foodeval.tasks import get_task, list_tasks, get_all_tasks
    >>> task = get_task("food_search")
    >>> task.task_type
    'retrieval'
    >>> list_tasks()
    ['bakery_match', 'beverage_match', 'concept_search', 'cross_lingual_match', 'cuisine_classify', 'diet_search', 'food_search', 'global_match', 'indian_match', 'noisy_menu_match', 'noisy_search', 'portion_size']
"""

from foodeval.tasks.base import BenchmarkTask, TaskResult
from foodeval.tasks.classification import ClassificationTask
from foodeval.tasks.pair_classification import PairClassificationTask
from foodeval.tasks.retrieval import RetrievalTask

__all__ = [
    "BenchmarkTask",
    "TaskResult",
    "RetrievalTask",
    "PairClassificationTask",
    "ClassificationTask",
    "TASK_REGISTRY",
    "get_task",
    "list_tasks",
    "get_all_tasks",
]

TASK_REGISTRY: dict[str, BenchmarkTask] = {
    "food_search": RetrievalTask("food_search"),
    "concept_search": RetrievalTask("concept_search"),
    "diet_search": RetrievalTask("diet_search"),
    "noisy_search": RetrievalTask("noisy_search"),
    "indian_match": PairClassificationTask("indian_match"),
    "global_match": PairClassificationTask("global_match"),
    "beverage_match": PairClassificationTask("beverage_match"),
    "bakery_match": PairClassificationTask("bakery_match"),
    "portion_size": PairClassificationTask("portion_size"),
    "noisy_menu_match": PairClassificationTask("noisy_menu_match"),
    "cross_lingual_match": PairClassificationTask("cross_lingual_match"),
    "cuisine_classify": ClassificationTask("cuisine_classify"),
}


def get_task(name: str) -> BenchmarkTask:
    """Look up a task by name.

    Args:
        name: Task identifier (e.g. "food_search", "indian_match").

    Returns:
        The corresponding BenchmarkTask instance.

    Raises:
        KeyError: If the task name is not in the registry.
    """
    if name not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY))
        raise KeyError(f"Unknown task {name!r}. Available: {available}")
    return TASK_REGISTRY[name]


def list_tasks() -> list[str]:
    """Return sorted list of all registered task names."""
    return sorted(TASK_REGISTRY)


def get_all_tasks() -> list[BenchmarkTask]:
    """Return all registered tasks in alphabetical order."""
    return [TASK_REGISTRY[name] for name in sorted(TASK_REGISTRY)]
