from __future__ import annotations

import re

from rag_app.chunking.classifier import ChunkProfile, classify_document
from rag_app.exceptions import InvalidContentError
from rag_app.models import ChunkRecord, ParsedDocument
from rag_app.parsing.cleaners import clean_text


class ChunkingService:
    SEPARATORS = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", ";", "，", ",", " ", ""]

    def __init__(
        self,
        *,
        normal_size: int = 1000,
        normal_overlap: int = 200,
        technical_size: int = 500,
        technical_overlap: int = 100,
    ) -> None:
        self.settings = {
            ChunkProfile.NORMAL: (normal_size, normal_overlap),
            ChunkProfile.TECHNICAL: (technical_size, technical_overlap),
        }
        for size, overlap in self.settings.values():
            if size < 1 or overlap < 0 or overlap >= size:
                raise ValueError("chunk overlap must satisfy 0 <= overlap < size")

    def _split_text(self, text: str, size: int, overlap: int) -> list[str]:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
                separators=self.SEPARATORS,
                length_function=len,
                keep_separator=True,
            )
            return splitter.split_text(text)
        except ImportError:
            # Dependency bootstrap fallback; production uses LangChain's splitter.
            parts = [
                part.strip()
                for part in re.split(r"(?<=[。！？.!?])|\n+", text)
                if part.strip() and re.search(r"[\w\u4e00-\u9fff]", part)
            ]
            chunks: list[str] = []
            current = ""
            for part in parts:
                if current and len(current) + len(part) > size:
                    chunks.append(current)
                    current = current[-overlap:] + part if overlap else part
                else:
                    current += part
            if current:
                chunks.append(current)
            return chunks

    def split(self, document: ParsedDocument, *, profile: ChunkProfile | str | None = None) -> list[ChunkRecord]:
        selected = ChunkProfile(profile) if profile else classify_document(document.text, document.doc_type)
        size, overlap = self.settings[selected]
        units = document.units or []
        sources = units or [type("Unit", (), {"text": document.text, "location": "body"})()]
        chunks: list[ChunkRecord] = []
        for unit in sources:
            for raw in self._split_text(unit.text, size, overlap):
                try:
                    content = clean_text(raw)
                except InvalidContentError:
                    continue
                chunks.append(
                    ChunkRecord.create(
                        source=document.source,
                        doc_type=document.doc_type,
                        content=content,
                        index=len(chunks),
                        location=unit.location,
                        chunk_profile=selected.value,
                    )
                )
        if not chunks:
            raise InvalidContentError("切片后未得到有效内容")
        return chunks
