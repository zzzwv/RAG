from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


@dataclass(slots=True)
class DocumentUnit:
    text: str
    location: str


@dataclass(slots=True)
class ParsedDocument:
    source: str
    doc_type: str
    text: str
    units: list[DocumentUnit] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    source: str
    doc_type: str
    content: str
    create_time: str
    location: str
    content_hash: str
    chunk_profile: str
    index: int

    @classmethod
    def create(
        cls,
        *,
        source: str,
        doc_type: str,
        content: str,
        index: int,
        location: str,
        chunk_profile: str,
        content_hash: str | None = None,
    ) -> "ChunkRecord":
        digest = content_hash or sha256(content.encode("utf-8")).hexdigest()
        identity = f"{source}\0{doc_type}\0{index}\0{digest}"
        chunk_id = sha256(identity.encode("utf-8")).hexdigest()
        return cls(
            chunk_id=chunk_id,
            source=source,
            doc_type=doc_type,
            content=content,
            create_time=datetime.now(timezone.utc).isoformat(),
            location=location,
            content_hash=digest,
            chunk_profile=chunk_profile,
            index=index,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "doc_source": self.source,
            "doc_type": self.doc_type,
            "create_time": self.create_time,
            "location": self.location,
            "content_hash": self.content_hash,
            "chunk_profile": self.chunk_profile,
            "chunk_index": self.index,
        }


@dataclass(slots=True)
class SearchHit:
    chunk: ChunkRecord
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    rank: int | None = None


@dataclass(slots=True)
class AnswerResult:
    answer: str
    references: list[SearchHit] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    status: str = "ok"
