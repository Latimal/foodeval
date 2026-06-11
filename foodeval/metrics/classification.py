"""Classification metrics (macro-averaged F1, accuracy, per-class report).

Usage:
    >>> from foodeval.metrics.classification import macro_f1, classification_report
    >>> macro_f1([0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 2, 2])
    0.8222...
    >>> report = classification_report([0, 1, 1], [0, 1, 0], label_names=["cat", "dog"])
    >>> report["per_class"]["dog"]["recall"]
    0.5
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
)


def macro_f1(
    y_true: list[int],
    y_pred: list[int],
    labels: list[int] | None = None,
) -> float:
    """Macro-averaged F1 across all classes.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        labels: Optional fixed class list to average over. When provided, the
            denominator is pinned to this set so the score is comparable across
            splits even when a class is absent from a particular fold. When
            None, sklearn averages over the union of labels present in the
            inputs.

    Returns 0.0 for empty inputs.
    """
    if not y_true or not y_pred:
        return 0.0
    return float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )


def macro_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Macro-averaged per-class accuracy (mean of per-class accuracies).

    For each class, accuracy = correct predictions for that class / total
    instances of that class. The macro average is the unweighted mean across
    classes.

    Returns 0.0 for empty inputs.
    """
    if not y_true or not y_pred:
        return 0.0

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    classes = np.unique(yt)

    if len(classes) == 0:
        return 0.0

    accs = []
    for c in classes:
        mask = yt == c
        if mask.sum() == 0:
            continue
        accs.append(float(np.sum(yp[mask] == c) / mask.sum()))

    return float(np.mean(accs)) if accs else 0.0


def classification_report(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str] | None = None,
) -> dict:
    """Per-class and aggregate classification metrics.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        label_names: Optional human-readable names for each label index. When
            provided, the report covers the full label set (every index in
            label_names), so the per-class breakdown and the headline macro_f1
            share the same denominator regardless of which classes appear in a
            particular split.

    Returns:
        Dict with keys:
            macro_f1: float
            macro_accuracy: float
            per_class: dict mapping label name -> {precision, recall, f1, support}
    """
    if not y_true or not y_pred:
        return {"macro_f1": 0.0, "macro_accuracy": 0.0, "per_class": {}}

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    # Pin the class list to the full fixed label set when names are given so
    # the per-class report and macro_f1 stay consistent across splits. Without
    # names, fall back to the classes observed in y_true.
    if label_names is None:
        classes = np.sort(np.unique(yt))
        label_names_map = {int(c): str(c) for c in classes}
    else:
        classes = np.arange(len(label_names))
        label_names_map = {i: name for i, name in enumerate(label_names)}

    prec, rec, f1s, sup = precision_recall_fscore_support(
        yt, yp, labels=classes, average=None, zero_division=0
    )

    per_class = {}
    for i, c in enumerate(classes):
        name = label_names_map[int(c)]
        per_class[name] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1s[i]),
            "support": int(sup[i]),
        }

    classes_list = [int(c) for c in classes]
    return {
        "macro_f1": macro_f1(y_true, y_pred, labels=classes_list),
        "macro_accuracy": macro_accuracy(y_true, y_pred),
        "per_class": per_class,
    }
