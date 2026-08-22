from __future__ import annotations

from rag_app.models import SearchHit


def profile_from_label(label: str) -> str | None:
    return {"自动判断": None, "普通文档": "normal", "技术文档": "technical"}.get(label)


def format_reference(hit: SearchHit) -> str:
    score = float(hit.rerank_score or 0.0)
    return f"{hit.chunk.source} · {hit.chunk.location} · 相关度 {score:.3f}"
