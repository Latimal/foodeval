"""Adapter wrapper for query/document prompt prefixes.

Some embedding families require asymmetric query/document prefixes or task
instructions. This wrapper keeps that concern out of task implementations while
making the prompt contract explicit in result metadata.
"""

from __future__ import annotations

import numpy as np

from foodeval.adapters.base import EmbeddingAdapter


class PromptedAdapter:
    """Wrap an adapter and prepend role-specific text before encoding."""

    def __init__(
        self,
        base: EmbeddingAdapter,
        query_prefix: str = "",
        document_prefix: str = "",
        text_prefix: str = "",
    ) -> None:
        self._base = base
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._text_prefix = text_prefix

    @staticmethod
    def _apply_prefix(texts: list[str], prefix: str) -> list[str]:
        if not prefix:
            return texts
        return [f"{prefix}{text}" for text in texts]

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        return self._base.encode(
            self._apply_prefix(texts, self._text_prefix),
            batch_size=batch_size,
            normalize=normalize,
        )

    def encode_queries(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        return self._base.encode(
            self._apply_prefix(texts, self._query_prefix),
            batch_size=batch_size,
            normalize=normalize,
        )

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        return self._base.encode(
            self._apply_prefix(texts, self._document_prefix),
            batch_size=batch_size,
            normalize=normalize,
        )

    @property
    def dimension(self) -> int:
        return self._base.dimension

    @property
    def name(self) -> str:
        suffixes: list[str] = []
        if self._query_prefix:
            suffixes.append("qprompt")
        if self._document_prefix:
            suffixes.append("dprompt")
        if self._text_prefix:
            suffixes.append("tprompt")
        if not suffixes:
            return self._base.name
        return f"{self._base.name}-{'-'.join(suffixes)}"
