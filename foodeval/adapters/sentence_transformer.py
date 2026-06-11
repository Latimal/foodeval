"""SentenceTransformer adapter for local or HuggingFace models.

Wraps any SentenceTransformer-compatible model (local path or HF model ID).
Supports Matryoshka dimension truncation and disk caching.

Usage:
    >>> from foodeval.adapters.sentence_transformer import SentenceTransformerAdapter
    >>> adapter = SentenceTransformerAdapter("BAAI/bge-m3", truncate_dim=384)  # doctest: +SKIP
    >>> embeddings = adapter.encode(["butter chicken", "paneer tikka"])  # doctest: +SKIP
    >>> embeddings.shape  # doctest: +SKIP
    (2, 384)
"""

from __future__ import annotations

import numpy as np

from foodeval.adapters.base import load_cache, save_cache

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]


class SentenceTransformerAdapter:
    """Adapter for sentence-transformers models.

    Args:
        model_name_or_path: HuggingFace model ID or local directory path.
        truncate_dim: If set, truncate embeddings to this dimension
            (Matryoshka). Must be <= the model's native dimension.
        device: PyTorch device string (e.g. "cuda", "cpu", "mps").
            If None, sentence-transformers picks automatically.
    """

    def __init__(
        self,
        model_name_or_path: str,
        truncate_dim: int | None = None,
        device: str | None = None,
    ) -> None:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerAdapter. "
                "Install it with: pip install 'foodeval[local]'"
            )

        kwargs: dict = {"trust_remote_code": True}
        if truncate_dim is not None:
            kwargs["truncate_dim"] = truncate_dim
        if device is not None:
            kwargs["device"] = device

        self._model = SentenceTransformer(model_name_or_path, **kwargs)
        self._model_name = model_name_or_path
        self._truncate_dim = truncate_dim

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: Input strings to encode.
            batch_size: Number of texts per forward pass.
            normalize: Whether to L2-normalize the output vectors.

        Returns:
            Float32 array of shape (N, D).
        """
        cached = load_cache(self.name, self.dimension, texts)
        if cached is not None:
            return cached

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        embeddings = embeddings.astype(np.float32)
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"{self._model_name} produced {embeddings.shape[1]}d "
                f"embeddings but adapter reports dimension {self.dimension}. "
                "A truncate_dim larger than the model's native width is "
                "not supported."
            )
        save_cache(self.name, self.dimension, texts, embeddings)
        return embeddings

    @property
    def dimension(self) -> int:
        """Embedding dimension (after truncation if applied)."""
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(
                f"Model {self._model_name} did not report an embedding dimension."
            )
        return int(dim)

    @property
    def name(self) -> str:
        """Human-readable model name for leaderboard display."""
        base = self._model_name.split("/")[-1]
        if self._truncate_dim is not None:
            return f"{base}-{self._truncate_dim}d"
        return base
