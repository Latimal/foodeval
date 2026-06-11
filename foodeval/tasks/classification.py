"""Classification task: embed items, train LogisticRegression, report macro F1.

Handles cuisine_classify. Items are encoded, then a LogisticRegression probe is
trained on an 80/20 stratified split. To reduce variance, the experiment is
repeated with multiple seeds and the mean/std are reported.

Usage:
    >>> from foodeval.tasks.classification import ClassificationTask
    >>> task = ClassificationTask("cuisine_classify")
    >>> result = task.run(adapter)  # doctest: +SKIP
    >>> print(f"macro_f1: {result.main_score:.4f}")  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

from foodeval.adapters.base import EmbeddingAdapter
from foodeval.metrics import macro_accuracy, macro_f1
from foodeval.metrics.classification import classification_report
from foodeval.tasks.base import BenchmarkTask, TaskResult


class ClassificationTask(BenchmarkTask):
    """Linear probe classification on top of frozen embeddings.

    Embeds all items once, then trains a LogisticRegression classifier on an
    80/20 stratified split. Repeats across multiple seeds to capture variance.
    The primary metric is the mean macro F1 across all seeds.
    """

    task_type = "classification"
    metric_name = "macro_f1"

    _N_SEEDS = 10
    _TEST_SIZE = 0.2
    _BASE_SEED = 42

    def __init__(self, task_name: str) -> None:
        super().__init__(task_name)
        self._items: list[dict[str, Any]] = []
        self._label_names: list[str] = []

    def load_data(self) -> None:
        """Load items and label names from the task JSON file.

        Validates that each item has text and a label, and that label_names
        covers all labels in the data.
        """
        data = self._load_json()
        self._data = data

        self._items = data["items"]
        self._label_names = data.get("label_names", [])

        if not self._items:
            raise ValueError(f"{self.name}: no items found")

        for item in self._items:
            if "text" not in item:
                raise ValueError(
                    f"{self.name}: item {item.get('id', '?')} missing 'text'"
                )
            if "label" not in item:
                raise ValueError(
                    f"{self.name}: item {item.get('id', '?')} missing 'label'"
                )

        # Validate labels are in label_names
        labels_in_data = {item["label"] for item in self._items}
        if self._label_names:
            missing = labels_in_data - set(self._label_names)
            if missing:
                raise ValueError(
                    f"{self.name}: labels in data not found in label_names: {missing}"
                )

    def evaluate(self, adapter: EmbeddingAdapter) -> TaskResult:
        """Encode items, train linear probes across seeds, aggregate results.

        Returns:
            TaskResult with mean macro F1, per-class breakdown from the
            canonical seed, accuracy, and cross-seed standard deviation.
        """
        if not self._items:
            raise RuntimeError(f"{self.name}: call load_data() before evaluate()")

        texts = [item["text"] for item in self._items]
        raw_labels = [item["label"] for item in self._items]

        # Map string labels to integers
        if self._label_names:
            label_to_idx = {name: i for i, name in enumerate(self._label_names)}
        else:
            unique_labels = sorted(set(raw_labels))
            label_to_idx = {name: i for i, name in enumerate(unique_labels)}
            self._label_names = unique_labels

        labels = np.array([label_to_idx[lb] for lb in raw_labels], dtype=np.int64)

        # Fixed class list spanning every label. Pinning the macro-F1
        # denominator to this set keeps per-seed scores comparable even when a
        # rare class is absent from a test split, and makes the headline score
        # agree with the per-class report.
        all_class_indices = list(range(len(self._label_names)))

        # Encode all items once
        embeddings = adapter.encode(texts, normalize=True)

        # Run multiple seeds
        seed_f1s: list[float] = []
        seed_accs: list[float] = []
        canonical_report: dict[str, Any] = {}
        canonical_predictions: list[dict[str, Any]] = []
        canonical_split: dict[str, list[int]] = {}

        for seed_offset in range(self._N_SEEDS):
            seed = self._BASE_SEED + seed_offset

            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=self._TEST_SIZE, random_state=seed
            )
            train_idx, test_idx = next(sss.split(embeddings, labels))

            X_train = embeddings[train_idx]
            X_test = embeddings[test_idx]
            y_train = labels[train_idx]
            y_test = labels[test_idx]

            clf = LogisticRegression(
                max_iter=1000,
                random_state=seed,
                solver="lbfgs",
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            f1 = macro_f1(y_test.tolist(), y_pred.tolist(), labels=all_class_indices)
            acc = macro_accuracy(y_test.tolist(), y_pred.tolist())
            seed_f1s.append(f1)
            seed_accs.append(acc)

            # Save the canonical seed (first) report for per-class details
            if seed_offset == 0:
                canonical_report = classification_report(
                    y_test.tolist(),
                    y_pred.tolist(),
                    label_names=self._label_names,
                )
                canonical_split = {
                    "train_idx": [int(i) for i in train_idx.tolist()],
                    "test_idx": [int(i) for i in test_idx.tolist()],
                }
                idx_to_label = {i: name for name, i in label_to_idx.items()}
                canonical_predictions = [
                    {
                        "id": self._items[int(item_idx)].get(
                            "id", f"{self.name}_{int(item_idx)}"
                        ),
                        "label": idx_to_label[int(true_label)],
                        "predicted": idx_to_label[int(pred_label)],
                    }
                    for item_idx, true_label, pred_label in zip(
                        test_idx, y_test, y_pred, strict=False
                    )
                ]

        mean_f1 = float(np.mean(seed_f1s))
        std_f1 = float(np.std(seed_f1s))
        mean_acc = float(np.mean(seed_accs))

        return TaskResult(
            task_name=self.name,
            main_score=round(mean_f1, 4),
            metric_name=self.metric_name,
            n_examples=len(self._items),
            details={
                "mean_macro_f1": round(mean_f1, 4),
                "std_macro_f1": round(std_f1, 4),
                "mean_accuracy": round(mean_acc, 4),
                "n_seeds": self._N_SEEDS,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_classes": len(self._label_names),
                "label_names": self._label_names,
                "per_class": canonical_report.get("per_class", {}),
                "per_seed_f1": [round(s, 4) for s in seed_f1s],
                "canonical_seed": self._BASE_SEED,
                "canonical_split": canonical_split,
                "canonical_predictions": canonical_predictions,
            },
        )
