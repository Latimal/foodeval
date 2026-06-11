"""Baseline matrix runner for FoodEval."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from foodeval.adapters.factory import build_adapter
from foodeval.adapters.prompted import PromptedAdapter
from foodeval.evaluate import BenchmarkResult, run_benchmark
from foodeval.preflight import contamination_report

PROTOCOL_VERSION = "foodeval-baseline-matrix-v1"


DEFAULT_MATRIX: list[dict[str, Any]] = [
    {
        "id": "bm25",
        "model": "bm25",
        "dimensions": [None],
        "tier": "lexical",
        "notes": "HashingVectorizer lexical baseline.",
    },
    {
        "id": "bge-m3",
        "model": "BAAI/bge-m3",
        "dimensions": [384, 512, 1024],
        "tier": "open_local",
        "notes": "Evaluate dense mode; separate sparse/multivector runs need a custom adapter.",
    },
    {
        "id": "multilingual-e5-large-instruct",
        "model": "intfloat/multilingual-e5-large-instruct",
        "dimensions": [384, 512, 1024],
        "tier": "open_local",
        "query_prefix": "Instruct: Retrieve relevant food/menu items.\nQuery: ",
        "document_prefix": "Passage: ",
        "notes": "Instruction/passage formatting per E5-style retrieval.",
    },
    {
        "id": "qwen3-embedding-0.6b",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "dimensions": [384, 512, 1024],
        "tier": "open_local",
        "query_prefix": "Instruct: Retrieve relevant food/menu items.\nQuery: ",
        "notes": "Instruction-aware query-side baseline.",
    },
    {
        "id": "qwen3-embedding-4b",
        "model": "Qwen/Qwen3-Embedding-4B",
        "dimensions": [512, 1024, 2560],
        "tier": "open_local_heavy",
        "query_prefix": "Instruct: Retrieve relevant food/menu items.\nQuery: ",
        "notes": "Heavy local/open baseline; run only with enough GPU/RAM.",
    },
    {
        "id": "qwen3-embedding-8b",
        "model": "Qwen/Qwen3-Embedding-8B",
        "dimensions": [512, 1024, 4096],
        "tier": "open_local_heavy",
        "query_prefix": "Instruct: Retrieve relevant food/menu items.\nQuery: ",
        "notes": "High-ceiling open baseline; expensive local run.",
    },
    {
        "id": "granite-embedding-311m-multilingual-r2",
        "model": "ibm-granite/granite-embedding-311m-multilingual-r2",
        "dimensions": [384, 512, 768],
        "tier": "open_local",
        "notes": "Governed multilingual open baseline.",
    },
    {
        "id": "jina-v5-text-small",
        "model": "jinaai/jina-embeddings-v5-text-small",
        "dimensions": [384, 512, 1024],
        "tier": "open_local",
        "query_prefix": "retrieval.query: ",
        "document_prefix": "retrieval.passage: ",
        "notes": "Use model-specific task prompts if the model card changes.",
    },
    {
        "id": "nomic-v2-moe",
        "model": "nomic-ai/nomic-embed-text-v2-moe",
        "dimensions": [256, 384, 768],
        "tier": "open_local",
        "query_prefix": "search_query: ",
        "document_prefix": "search_document: ",
        "notes": "MRL-capable lightweight baseline.",
    },
    {
        "id": "openai-te3-large",
        "model": "openai:text-embedding-3-large",
        "dimensions": [384, 512, 1024, 3072],
        "tier": "api",
        "required_env": ["OPENAI_API_KEY"],
        "notes": "OpenAI direct API; no asymmetric query/document input mode.",
    },
    {
        "id": "openai-te3-small",
        "model": "openai:text-embedding-3-small",
        "dimensions": [384, 512, 1536],
        "tier": "api",
        "required_env": ["OPENAI_API_KEY"],
        "notes": "OpenAI direct API; no asymmetric query/document input mode.",
    },
    {
        "id": "cohere-embed-v4",
        "model": "cohere:embed-v4.0",
        "dimensions": [384, 512, 1024, 1536],
        "tier": "api",
        "required_env": ["COHERE_API_KEY"],
        "notes": "Adapter uses search_query/search_document roles.",
    },
    {
        "id": "gemini-embedding-2",
        "model": "gemini:gemini-embedding-2",
        "dimensions": [384, 512, 1024, 3072],
        "tier": "api",
        "required_env": ["GEMINI_API_KEY"],
        "notes": "Gemini API direct; adapter uses RETRIEVAL_QUERY/RETRIEVAL_DOCUMENT task types.",
    },
    {
        "id": "gemini-embedding-001",
        "model": "gemini:gemini-embedding-001",
        "dimensions": [384, 512, 1024, 3072],
        "tier": "api",
        "required_env": ["GEMINI_API_KEY"],
        "notes": "Gemini API direct; adapter uses RETRIEVAL_QUERY/RETRIEVAL_DOCUMENT task types.",
    },
    {
        "id": "vertex-gemini-embedding-001",
        "model": "vertex:gemini-embedding-001",
        "dimensions": [384, 512, 1024, 3072],
        "tier": "api",
        "required_env": ["GOOGLE_CLOUD_PROJECT"],
        "notes": "Vertex AI publisher endpoint; ADC auth (gcloud application-default login). RETRIEVAL_QUERY/RETRIEVAL_DOCUMENT task types.",
    },
    {
        "id": "vertex-gemini-embedding-2",
        "model": "vertex:gemini-embedding-2",
        "dimensions": [384, 512, 1024, 3072],
        "tier": "api",
        "required_env": ["GOOGLE_CLOUD_PROJECT"],
        "notes": "Vertex AI embedContent surface (no batch; threaded singles); ADC auth. RETRIEVAL_QUERY/RETRIEVAL_DOCUMENT task types.",
    },
    {
        "id": "voyage-4-large",
        "model": "voyage:voyage-4-large",
        "dimensions": [384, 512, 1024, 2048],
        "tier": "api",
        "required_env": ["VOYAGE_API_KEY"],
        "notes": "Adapter uses query/document roles.",
    },
    {
        "id": "voyage-4",
        "model": "voyage:voyage-4",
        "dimensions": [384, 512, 1024, 2048],
        "tier": "api",
        "required_env": ["VOYAGE_API_KEY"],
        "notes": "Adapter uses query/document roles.",
    },
    {
        "id": "bedrock-cohere-embed-v4",
        "model": "bedrock:cohere.embed-v4:0",
        "dimensions": [384, 512, 1024, 1536],
        "tier": "bedrock_api",
        "required_env_any": ["AWS_ACCESS_KEY_ID", "AWS_PROFILE"],
        "notes": "Bedrock Cohere baseline; uses search_query/search_document roles.",
    },
]


@dataclass
class MatrixRun:
    entry: dict[str, Any]
    dimension: int | None

    @property
    def id(self) -> str:
        dim = "native" if self.dimension is None else str(self.dimension)
        suffix = dim if self.dimension is None else f"{dim}d"
        return f"{self.entry['id']}-{suffix}"


def load_matrix(path: str | None = None) -> list[dict[str, Any]]:
    """Load a matrix config JSON or return the built-in default."""
    if path is None:
        return DEFAULT_MATRIX
    with open(Path(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("models", [])
    if isinstance(data, list):
        return data
    raise ValueError("Matrix config must be a list or an object with a 'models' key.")


def load_env_files(paths: list[str]) -> list[str]:
    """Load KEY=VALUE pairs from env files without printing secret values."""
    loaded: list[str] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
        loaded.append(str(path))
    return loaded


def matrix_runs(
    matrix: list[dict[str, Any]],
    model_ids: set[str] | None = None,
    dims: set[int] | None = None,
) -> list[MatrixRun]:
    """Expand model entries into model/dimension run records."""
    runs: list[MatrixRun] = []
    for entry in matrix:
        if model_ids and entry.get("id") not in model_ids:
            continue
        for dim in entry.get("dimensions", [None]):
            if dims is not None and dim is not None and int(dim) not in dims:
                continue
            runs.append(MatrixRun(entry=entry, dimension=dim))
    return runs


def missing_env(entry: dict[str, Any]) -> list[str]:
    """Return missing environment variables for a matrix entry."""
    missing = [key for key in entry.get("required_env", []) if not os.environ.get(key)]
    any_group = entry.get("required_env_any", [])
    if any_group and not any(os.environ.get(key) for key in any_group):
        missing.append(" or ".join(any_group))
    return missing


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - metadata best-effort
        return None


def _wrap_prompts(adapter, entry: dict[str, Any]):
    query_prefix = entry.get("query_prefix", "")
    document_prefix = entry.get("document_prefix", "")
    text_prefix = entry.get("text_prefix", "")
    if not (query_prefix or document_prefix or text_prefix):
        return adapter
    return PromptedAdapter(
        adapter,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        text_prefix=text_prefix,
    )


def run_matrix(
    output_dir: str,
    tasks: list[str] | None = None,
    config_path: str | None = None,
    model_ids: set[str] | None = None,
    dims: set[int] | None = None,
    compare_paths: list[str] | None = None,
    env_files: list[str] | None = None,
    execute: bool = False,
    skip_missing_env: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Plan or execute a baseline matrix."""
    loaded_env_files = load_env_files(env_files or [])
    matrix = load_matrix(config_path)
    runs = matrix_runs(matrix, model_ids=model_ids, dims=dims)
    preflight = contamination_report(compare_paths or [], task_names=tasks)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "preflight.json", "w", encoding="utf-8") as f:
        json.dump(preflight, f, indent=2, ensure_ascii=False)

    plan: list[dict[str, Any]] = []
    results: list[str] = []
    skipped: list[dict[str, Any]] = []

    for run in runs:
        missing = missing_env(run.entry)
        record = {
            "id": run.id,
            "model_id": run.entry.get("id"),
            "model": run.entry.get("model"),
            "dimension": run.dimension,
            "tier": run.entry.get("tier", "unknown"),
            "missing_env": missing,
            "will_execute": execute and not (missing and skip_missing_env),
        }
        plan.append(record)

        if not execute:
            continue
        if missing and skip_missing_env:
            skipped.append(record)
            continue

        adapter = build_adapter(str(run.entry["model"]), run.dimension)
        adapter = _wrap_prompts(adapter, run.entry)
        metadata = {
            "protocol": PROTOCOL_VERSION,
            "git_sha": _git_sha(),
            "matrix_entry": run.entry,
            "dimension": run.dimension,
            "task_manifest": preflight["tasks"],
            "contamination_summary": preflight["summary"],
            "prompt_contract": {
                "query_prefix": run.entry.get("query_prefix", ""),
                "document_prefix": run.entry.get("document_prefix", ""),
                "text_prefix": run.entry.get("text_prefix", ""),
            },
        }
        result: BenchmarkResult = run_benchmark(
            adapter=adapter,
            tasks=tasks,
            verbose=verbose,
            metadata=metadata,
        )
        output_path = out_dir / f"{run.id}.json"
        result.to_json(str(output_path))
        results.append(str(output_path))

    summary = {
        "protocol": PROTOCOL_VERSION,
        "execute": execute,
        "output_dir": str(out_dir),
        "planned": plan,
        "skipped": skipped,
        "results": results,
        "preflight": str(out_dir / "preflight.json"),
        "loaded_env_files": loaded_env_files,
    }
    with open(out_dir / "matrix-plan.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def write_default_matrix(path: str) -> None:
    """Write the built-in default matrix to JSON for editing."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"models": DEFAULT_MATRIX}, f, indent=2, ensure_ascii=False)
