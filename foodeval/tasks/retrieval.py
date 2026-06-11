"""Retrieval task: encode queries and corpus, rank by cosine, score NDCG@10.

Handles food_search, concept_search, diet_search, and noisy_search. All four
share the same data format and evaluation logic.

Usage:
    >>> from foodeval.tasks.retrieval import RetrievalTask
    >>> task = RetrievalTask("food_search")
    >>> result = task.run(adapter)  # doctest: +SKIP
    >>> print(f"{result.metric_name}: {result.main_score:.4f}")  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any

import numpy as np

from foodeval.adapters.base import EmbeddingAdapter, encode_documents, encode_queries
from foodeval.metrics import bootstrap_ci, ndcg_at_k
from foodeval.tasks.base import BenchmarkTask, TaskResult


class RetrievalTask(BenchmarkTask):
    """Ranked retrieval evaluation via cosine similarity and NDCG@k.

    For each query, all corpus items are ranked by cosine similarity against
    the query embedding. NDCG@k is computed using the graded relevance
    judgments from the data file. The primary metric is the mean NDCG@k
    across all queries.
    """

    task_type = "retrieval"
    metric_name = "ndcg@10"

    def __init__(self, task_name: str, k: int = 10) -> None:
        super().__init__(task_name)
        self._k = k
        self._corpus: list[str] = []
        self._queries: list[dict[str, Any]] = []

    def load_data(self) -> None:
        """Load corpus and queries from the task JSON file.

        Validates that the expected keys are present and that relevance
        judgments reference items that exist in the corpus.
        """
        data = self._load_json()
        self._data = data

        self._corpus = data["corpus"]
        self._queries = data["queries"]

        if not self._corpus:
            raise ValueError(f"{self.name}: corpus is empty")
        if not self._queries:
            raise ValueError(f"{self.name}: no queries found")

        corpus_set = set(self._corpus)
        for q in self._queries:
            unknown = [
                item for item in q.get("relevance", {}) if item not in corpus_set
            ]
            if unknown:
                raise ValueError(
                    f"{self.name}: query {q['id']} references items not in corpus: "
                    f"{unknown[:3]}"
                )

    def evaluate(self, adapter: EmbeddingAdapter) -> TaskResult:
        """Encode corpus and queries, rank, compute NDCG@k.

        Returns:
            TaskResult with mean NDCG@k, per-domain breakdown, per-query
            scores, and bootstrap confidence interval.
        """
        if not self._corpus or not self._queries:
            raise RuntimeError(f"{self.name}: call load_data() before evaluate()")

        # Encode everything
        corpus_embeddings = encode_documents(adapter, self._corpus, normalize=True)
        query_texts = [q["query"] for q in self._queries]
        query_embeddings = encode_queries(adapter, query_texts, normalize=True)

        per_query_scores: list[float] = []
        per_domain: dict[str, list[float]] = {}
        per_query_details: list[dict[str, Any]] = []

        for qi, q in enumerate(self._queries):
            q_emb = query_embeddings[qi : qi + 1]  # (1, D)
            # Cosine similarity (vectors are already normalized)
            sims = (q_emb @ corpus_embeddings.T).flatten()  # (N_corpus,)

            # Rank corpus items by descending similarity. Stable sort keeps
            # tie-order reproducible across numpy versions.
            ranked_indices = np.argsort(-sims, kind="stable")

            # Build relevance list over ALL ranked items so IDCG reflects the
            # full relevance set. ndcg_at_k truncates DCG at k internally and
            # derives IDCG from this complete list; slicing to top-k here would
            # inflate NDCG when relevant items rank beyond position k.
            relevance_map = q.get("relevance", {})
            relevance_at_rank = [
                relevance_map.get(self._corpus[idx], 0) for idx in ranked_indices
            ]

            score = ndcg_at_k(relevance_at_rank, k=self._k)
            per_query_scores.append(score)

            domain = q.get("domain", "unknown")
            per_domain.setdefault(domain, []).append(score)

            per_query_details.append(
                {
                    "id": q["id"],
                    "domain": domain,
                    "query": q["query"],
                    "ndcg": round(score, 4),
                    "top_k": [
                        {
                            "rank": rank + 1,
                            "text": self._corpus[idx],
                            "score": round(float(sims[idx]), 6),
                            "relevance": relevance_map.get(self._corpus[idx], 0),
                        }
                        for rank, idx in enumerate(ranked_indices[: self._k])
                    ],
                }
            )

        mean_score = float(np.mean(per_query_scores))
        ci = bootstrap_ci(per_query_scores)

        domain_summary = {
            domain: {
                "mean_ndcg": round(float(np.mean(scores)), 4),
                "n_queries": len(scores),
            }
            for domain, scores in sorted(per_domain.items())
        }

        return TaskResult(
            task_name=self.name,
            main_score=round(mean_score, 4),
            metric_name=self.metric_name,
            n_examples=len(self._queries),
            details={
                "k": self._k,
                "n_corpus": len(self._corpus),
                "per_domain": domain_summary,
                "confidence_interval": ci,
                "per_query": per_query_details,
            },
        )
