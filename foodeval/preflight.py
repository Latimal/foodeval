"""Benchmark manifest and contamination preflight utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from foodeval.tasks import get_task, list_tasks

_TEXT_EXTENSIONS = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".md"}
_MAX_EXAMPLES = 25


def normalize_text(text: str) -> str:
    """Normalize text for exact contamination scans."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[\W_]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def file_sha256(path: Path) -> str:
    """Return a SHA-256 digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class BenchmarkSurface:
    texts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    pairs: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict)
    query_document_edges: dict[tuple[str, str], list[dict[str, Any]]] = field(
        default_factory=dict
    )
    manifest: dict[str, dict[str, Any]] = field(default_factory=dict)


def _add_text(
    surface: BenchmarkSurface,
    task_name: str,
    kind: str,
    item_id: str,
    text: str,
) -> None:
    key = normalize_text(text)
    if not key:
        return
    surface.texts.setdefault(key, []).append(
        {"task": task_name, "kind": kind, "id": item_id, "text": text}
    )


def _add_pair(
    pairs: dict[tuple[str, str], list[dict[str, Any]]],
    left: str,
    right: str,
    info: dict[str, Any],
) -> None:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return
    key = tuple(sorted((left_norm, right_norm)))
    pairs.setdefault(key, []).append(info)


def collect_benchmark_surface(task_names: list[str] | None = None) -> BenchmarkSurface:
    """Collect benchmark text, pair, and query-document surfaces."""
    names = task_names or list_tasks()
    surface = BenchmarkSurface()

    for task_name in names:
        task = get_task(task_name)
        path = task.data_path
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        manifest: dict[str, Any] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "version": data.get("version", "unknown"),
            "metadata": data.get("metadata", {}),
        }

        if "corpus" in data and "queries" in data:
            corpus = data.get("corpus", [])
            queries = data.get("queries", [])
            manifest.update(
                {
                    "type": "retrieval",
                    "n_corpus": len(corpus),
                    "n_queries": len(queries),
                }
            )
            for i, item in enumerate(corpus):
                _add_text(surface, task_name, "corpus", f"corpus_{i}", item)
            for query in queries:
                qid = str(query.get("id", "query"))
                qtext = str(query.get("query", ""))
                _add_text(surface, task_name, "query", qid, qtext)
                for document in query.get("relevance", {}):
                    _add_pair(
                        surface.query_document_edges,
                        qtext,
                        str(document),
                        {
                            "task": task_name,
                            "query_id": qid,
                            "query": qtext,
                            "document": document,
                        },
                    )
        elif "pairs" in data:
            pairs = data.get("pairs", [])
            manifest.update({"type": "pair_classification", "n_pairs": len(pairs)})
            for i, pair in enumerate(pairs):
                pair_id = str(pair.get("id", f"pair_{i}"))
                text_a = str(pair.get("text_a", ""))
                text_b = str(pair.get("text_b", ""))
                _add_text(surface, task_name, "pair_text_a", pair_id, text_a)
                _add_text(surface, task_name, "pair_text_b", pair_id, text_b)
                _add_pair(
                    surface.pairs,
                    text_a,
                    text_b,
                    {
                        "task": task_name,
                        "id": pair_id,
                        "label": pair.get("label"),
                        "text_a": text_a,
                        "text_b": text_b,
                    },
                )
        elif "items" in data:
            items = data.get("items", [])
            manifest.update({"type": "classification", "n_items": len(items)})
            for i, item in enumerate(items):
                _add_text(
                    surface,
                    task_name,
                    "classification_text",
                    str(item.get("id", f"item_{i}")),
                    str(item.get("text", "")),
                )
        else:
            manifest["type"] = "unknown"

        surface.manifest[task_name] = manifest

    return surface


def _iter_source_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            files.extend(
                p
                for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix.lower() in _TEXT_EXTENSIONS
            )
        elif path.is_file() and path.suffix.lower() in _TEXT_EXTENSIONS:
            files.append(path)
    return files


def _walk_json_strings(value: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(value, str):
        strings.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            strings.extend(_walk_json_strings(item))
    elif isinstance(value, list):
        for item in value:
            strings.extend(_walk_json_strings(item))
    return strings


def _walk_json_pairs(value: Any) -> list[tuple[str, str, dict[str, Any]]]:
    pairs: list[tuple[str, str, dict[str, Any]]] = []
    if isinstance(value, dict):
        if "text_a" in value and "text_b" in value:
            pairs.append(
                (
                    str(value.get("text_a", "")),
                    str(value.get("text_b", "")),
                    {"label": value.get("label")},
                )
            )
        for item in value.values():
            pairs.extend(_walk_json_pairs(item))
    elif isinstance(value, list):
        for item in value:
            pairs.extend(_walk_json_pairs(item))
    return pairs


def _extract_source(
    path: Path,
) -> tuple[list[str], list[tuple[str, str, dict[str, Any]]]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _walk_json_strings(data), _walk_json_pairs(data)

    if suffix == ".jsonl":
        strings: list[str] = []
        pairs: list[tuple[str, str, dict[str, Any]]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                strings.extend(_walk_json_strings(data))
                pairs.extend(_walk_json_pairs(data))
        return strings, pairs

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        strings = []
        pairs = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                strings.extend(str(v) for v in row.values() if v is not None)
                if "text_a" in row and "text_b" in row:
                    pairs.append(
                        (
                            str(row.get("text_a", "")),
                            str(row.get("text_b", "")),
                            {"label": row.get("label")},
                        )
                    )
        return strings, pairs

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [line.strip() for line in f if line.strip()], []


def contamination_report(
    compare_paths: list[str],
    task_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compare benchmark surfaces against external files."""
    surface = collect_benchmark_surface(task_names)
    report: dict[str, Any] = {
        "normalization": "NFKC + casefold + non-word collapse",
        "tasks": surface.manifest,
        "benchmark_surface": {
            "n_unique_texts": len(surface.texts),
            "n_unique_pairs": len(surface.pairs),
            "n_unique_query_document_edges": len(surface.query_document_edges),
        },
        "comparisons": [],
    }

    total_text_hits = 0
    total_pair_hits = 0
    total_edge_hits = 0

    for path in _iter_source_files(compare_paths):
        try:
            strings, pairs = _extract_source(path)
        except Exception as exc:  # noqa: BLE001 - report and continue
            report["comparisons"].append({"path": str(path), "error": str(exc)})
            continue

        source_texts = {normalize_text(s) for s in strings if normalize_text(s)}
        text_hits = sorted(source_texts & set(surface.texts))

        source_pairs: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for left, right, info in pairs:
            _add_pair(source_pairs, left, right, info)
        pair_hits = sorted(set(source_pairs) & set(surface.pairs))
        edge_hits = sorted(set(source_pairs) & set(surface.query_document_edges))

        total_text_hits += len(text_hits)
        total_pair_hits += len(pair_hits)
        total_edge_hits += len(edge_hits)

        report["comparisons"].append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "n_source_texts": len(source_texts),
                "n_source_pairs": len(source_pairs),
                "exact_text_overlap_count": len(text_hits),
                "pair_overlap_count": len(pair_hits),
                "query_document_edge_overlap_count": len(edge_hits),
                "exact_text_overlap_examples": [
                    {"normalized": hit, "benchmark": surface.texts[hit][0]}
                    for hit in text_hits[:_MAX_EXAMPLES]
                ],
                "pair_overlap_examples": [
                    {"normalized_pair": list(hit), "benchmark": surface.pairs[hit][0]}
                    for hit in pair_hits[:_MAX_EXAMPLES]
                ],
                "query_document_edge_overlap_examples": [
                    {
                        "normalized_pair": list(hit),
                        "benchmark": surface.query_document_edges[hit][0],
                    }
                    for hit in edge_hits[:_MAX_EXAMPLES]
                ],
            }
        )

    report["summary"] = {
        "n_compared_files": len(report["comparisons"]),
        "total_exact_text_overlaps": total_text_hits,
        "total_pair_overlaps": total_pair_hits,
        "total_query_document_edge_overlaps": total_edge_hits,
    }
    return report


def write_report(report: dict[str, Any], output_path: str) -> None:
    """Write a preflight report JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
