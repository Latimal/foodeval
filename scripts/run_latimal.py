#!/usr/bin/env python3
"""Reproduce the Latimal food-embed-v1 leaderboard row via the public API.

Usage:
    export LATIMAL_API_KEY=...   # get a key at https://latimal.com
    python3 scripts/run_latimal.py --dim 384 --output results/latimal_food_embed_v1_384.json

Wraps the production /embed endpoint (https://dish-embed.latimal.com) in the
standard EmbeddingAdapter protocol and runs the full FoodEval suite. This is
the same custom-adapter route CONTRIBUTING.md asks of any submission whose
model has no built-in adapter.

One caveat, documented at https://dish-embed.latimal.com/docs: /embed applies
the API's standard input preprocessing (noise and spelling normalization)
before encoding. The published leaderboard row was measured on raw text like
every other row, so scores from this script land at or above the published
numbers wherever inputs carry markup noise; a verification run measured the
aggregate +0.012 above the published row, with task deltas from -0.004 to
+0.037.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


DEFAULT_BASE_URL = "https://dish-embed.latimal.com"
API_MAX_ITEMS = 512
MAX_RETRIES = 5


class LatimalAPIAdapter:
    """Adapter for the Latimal production /embed endpoint."""

    def __init__(
        self,
        dimension: int = 384,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._dimension = dimension
        self._api_key = api_key or os.environ.get("LATIMAL_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "Latimal API key required. Set LATIMAL_API_KEY env var "
                "(keys at https://latimal.com)."
            )
        self._base_url = (base_url or os.environ.get("LATIMAL_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")

    @property
    def name(self) -> str:
        return "Latimal food-embed-v1"

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ):
        import numpy as np
        out: list[list[float]] = []
        step = min(batch_size, API_MAX_ITEMS)
        for i in range(0, len(texts), step):
            out.extend(self._embed_batch(texts[i : i + step]))
        if len(out) != len(texts):
            raise RuntimeError(
                f"API returned {len(out)} embeddings for {len(texts)} inputs"
            )
        arr = np.asarray(out, dtype=np.float32)
        if arr.shape[1] != self._dimension:
            raise RuntimeError(
                f"API returned {arr.shape[1]}-d vectors, expected {self._dimension}"
            )
        if normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr

    def _embed_batch(self, items: list[str]) -> list[list[float]]:
        payload = json.dumps({"items": items, "dimension": self._dimension}).encode("utf-8")
        for attempt in range(MAX_RETRIES):
            req = urllib.request.Request(
                f"{self._base_url}/embed",
                data=payload,
                method="POST",
            )
            req.add_header("Content-Type", "application/json")
            req.add_header("X-API-Key", self._api_key)
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read().decode("utf-8"))["embeddings"]
            except urllib.error.HTTPError as exc:
                # Retry rate limits and transient server errors with backoff.
                if exc.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                    time.sleep(2**attempt)
                    continue
                body = exc.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"API error {exc.code}: {body}") from exc
            except urllib.error.URLError:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2**attempt)
                    continue
                raise
        raise RuntimeError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run FoodEval against the Latimal production API."
    )
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension (default 384)")
    parser.add_argument("--output", default=None, help="Path to write the result JSON")
    parser.add_argument("--tasks", default=None, help="Comma-separated task subset (default: all 12)")
    args = parser.parse_args()

    try:
        from foodeval.evaluate import run_benchmark
    except ImportError:
        print(
            'foodeval is not installed in this environment. From the repo root: pip install -e ".[all]"',
            file=sys.stderr,
        )
        return 1

    try:
        adapter = LatimalAPIAdapter(dimension=args.dim)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    tasks = [s.strip() for s in args.tasks.split(",")] if args.tasks else None
    result = run_benchmark(adapter, tasks=tasks)
    print(result.to_markdown())
    if args.output:
        result.to_json(args.output)
        print(f"\nWrote {args.output}", file=sys.stderr)
    errored = [name for name, tr in result.task_results.items() if tr.errored]
    if errored:
        print(
            f"{len(errored)} task(s) errored: {', '.join(errored)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
