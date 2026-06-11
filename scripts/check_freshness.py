#!/usr/bin/env python3
"""Verify every result JSON reflects the CURRENT benchmark data.

For each result file in results/, check that:
  1. n_examples for each task matches the current data file's example count.
  2. cuisine_classify carries the current class count (details.n_classes and
     len(details.label_names)) and that it equals the current data's class count.
  3. retrieval tasks' details.n_corpus matches current corpus size.

Exits 0 if all result files are fresh, 1 otherwise. Prints a per-file,
per-task report of any drift.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "foodeval" / "data"
RESULTS_DIR = ROOT / "results"


def data_counts() -> dict[str, dict]:
    """Compute current n_examples / n_corpus / n_classes per task from data/."""
    counts: dict[str, dict] = {}
    for f in sorted(DATA_DIR.glob("*.json")):
        d = json.load(open(f))
        task = f.stem
        rec: dict = {}
        if "pairs" in d:
            rec["n_examples"] = len(d["pairs"])
        elif "items" in d:
            rec["n_examples"] = len(d["items"])
            labels = {i["label"] for i in d["items"]}
            rec["n_classes"] = len(labels)
            rec["label_names_len"] = len(d.get("label_names", []))
        elif "queries" in d:
            rec["n_examples"] = len(d["queries"])
            rec["n_corpus"] = len(d.get("corpus", []))
        counts[task] = rec
    return counts


def main() -> int:
    current = data_counts()
    result_files = sorted(RESULTS_DIR.glob("*.json"))
    if not result_files:
        print("No result files found.")
        return 1

    print(f"Current data counts ({len(current)} tasks):")
    for t in sorted(current):
        print(f"  {t:24s} {current[t]}")
    print()

    any_stale = False
    for rf in result_files:
        res = json.load(open(rf))
        tasks = res.get("tasks", {})
        problems: list[str] = []

        # Every current task must be present in the result and match counts.
        for t, exp in current.items():
            if t not in tasks:
                problems.append(f"MISSING task {t}")
                continue
            tr = tasks[t]
            det = tr.get("details", {})
            # n_examples
            if tr.get("n_examples") != exp.get("n_examples"):
                problems.append(
                    f"{t}.n_examples {tr.get('n_examples')} != data {exp.get('n_examples')}"
                )
            # corpus size for retrieval
            if "n_corpus" in exp and det.get("n_corpus") != exp["n_corpus"]:
                problems.append(
                    f"{t}.details.n_corpus {det.get('n_corpus')} != data {exp['n_corpus']}"
                )
            # class count for classification
            if "n_classes" in exp:
                if det.get("n_classes") != exp["n_classes"]:
                    problems.append(
                        f"{t}.details.n_classes {det.get('n_classes')} != data {exp['n_classes']}"
                    )
                if len(det.get("label_names", [])) != exp["label_names_len"]:
                    problems.append(
                        f"{t}.details.label_names len {len(det.get('label_names', []))} "
                        f"!= data {exp['label_names_len']}"
                    )

        # Extra tasks in result that no longer exist in data
        for t in tasks:
            if t not in current:
                problems.append(f"EXTRA task {t} not in current data")

        status = "STALE" if problems else "FRESH"
        if problems:
            any_stale = True
        print(f"[{status}] {rf.name}")
        for p in problems:
            print(f"      - {p}")

    print()
    if any_stale:
        print("RESULT: at least one file is STALE")
        return 1
    print("RESULT: all result files are FRESH against current data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
