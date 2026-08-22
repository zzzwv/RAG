from __future__ import annotations

from rag_app.models import SearchHit


def reciprocal_rank_fusion(
    rankings: list[list[SearchHit]], *, k: int = 60, limit: int = 10
) -> list[SearchHit]:
    if k <= 0 or limit <= 0:
        raise ValueError("k and limit must be positive")
    by_hash: dict[str, SearchHit] = {}
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen_in_ranking: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            key = item.chunk.content_hash
            if key in seen_in_ranking:
                continue
            seen_in_ranking.add(key)
            by_hash.setdefault(key, item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    fused = []
    for key, item in by_hash.items():
        item.rrf_score = scores[key]
        fused.append(item)
    fused.sort(key=lambda item: (-item.rrf_score, item.chunk.chunk_id))
    for rank, item in enumerate(fused[:limit], start=1):
        item.rank = rank
    return fused[:limit]
