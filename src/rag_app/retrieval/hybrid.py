from __future__ import annotations

from rag_app.models import SearchHit
from rag_app.retrieval.rrf import reciprocal_rank_fusion


class HybridRetriever:
    def __init__(
        self,
        vector_store,
        bm25_store,
        *,
        vector_top_k: int = 10,
        bm25_top_k: int = 10,
        fused_limit: int = 10,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.fused_limit = fused_limit
        self.rrf_k = rrf_k

    def search(self, query: str) -> list[SearchHit]:
        vector_hits = self.vector_store.search(query, top_k=self.vector_top_k)
        keyword_hits = self.bm25_store.search(query, top_k=self.bm25_top_k)
        return reciprocal_rank_fusion(
            [vector_hits, keyword_hits], k=self.rrf_k, limit=self.fused_limit
        )
