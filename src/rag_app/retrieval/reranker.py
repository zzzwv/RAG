from __future__ import annotations

import math

from rag_app.models import SearchHit


class BGEReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        *,
        model=None,
        threshold: float = 0.35,
        top_n: int = 3,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model
        self.threshold = threshold
        self.top_n = top_n
        self.device = device

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device, max_length=512)
        return self._model

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exponent = math.exp(value)
        return exponent / (1.0 + exponent)

    @staticmethod
    def _identity(value):
        return value

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        pairs = [(query, hit.chunk.content) for hit in hits]
        raw_scores = self._load().predict(
            pairs, show_progress_bar=False, activation_fn=self._identity
        )
        kept: list[SearchHit] = []
        for hit, raw_score in zip(hits, raw_scores):
            hit.rerank_score = self._sigmoid(float(raw_score))
            if hit.rerank_score >= self.threshold:
                kept.append(hit)
        kept.sort(key=lambda item: (-float(item.rerank_score or 0.0), item.chunk.chunk_id))
        for rank, hit in enumerate(kept[: self.top_n], start=1):
            hit.rank = rank
        return kept[: self.top_n]
