from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from rag_app.models import ChunkRecord, SearchHit


class BM25Store:
    def __init__(self, tokenizer: Callable[[str], list[str]] | None = None, bm25_factory=None) -> None:
        if tokenizer is None:
            import jieba

            def jieba_tokenize(text: str) -> list[str]:
                return [token.strip() for token in jieba.lcut(text) if token.strip()]

            tokenizer = jieba_tokenize
        if bm25_factory is None:
            from rank_bm25 import BM25Okapi

            bm25_factory = BM25Okapi
        self.tokenizer = tokenizer
        self.bm25_factory = bm25_factory
        self._chunks: list[ChunkRecord] = []
        self._index: Any | None = None
        self._lock = threading.RLock()

    def rebuild(self, chunks: list[ChunkRecord]) -> None:
        with self._lock:
            self._chunks = list(chunks)
            corpus = [self.tokenizer(chunk.content) for chunk in self._chunks]
            self._index = self.bm25_factory(corpus) if corpus else None

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        if top_k < 1:
            return []
        with self._lock:
            if self._index is None:
                return []
            scores = self._index.get_scores(self.tokenizer(query))
            ranked = sorted(enumerate(scores), key=lambda pair: (-float(pair[1]), pair[0]))[:top_k]
            return [
                SearchHit(chunk=self._chunks[index], bm25_score=float(score))
                for index, score in ranked
                if float(score) > 0
            ]
