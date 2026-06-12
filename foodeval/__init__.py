"""FoodEval: The First Benchmark for Food Domain Text Embeddings.

FoodEval evaluates embedding models on 12 food-specific tasks spanning
retrieval, deduplication, cross-lingual matching, and cuisine classification.

Quick start:
    >>> from foodeval.evaluate import run_benchmark
    >>> from foodeval.adapters.sentence_transformer import SentenceTransformerAdapter
    >>> adapter = SentenceTransformerAdapter("BAAI/bge-m3", truncate_dim=384)  # doctest: +SKIP
    >>> result = run_benchmark(adapter, tasks=["food_search"])  # doctest: +SKIP
    >>> print(result.aggregate_score)  # doctest: +SKIP
"""

__version__ = "0.1.1"
