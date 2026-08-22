from __future__ import annotations

from pathlib import Path

from rag_app.exceptions import ValidationError
from rag_app.models import ParsedDocument
from rag_app.parsing.pdf_parser import PDFParser
from rag_app.parsing.text_parser import TextParser
from rag_app.parsing.web_parser import WebParser
from rag_app.parsing.word_parser import WordParser


class DocumentParsingPipeline:
    ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".md", ".txt"}

    def __init__(self, *, max_file_size: int = 20 * 1024 * 1024, web_parser: WebParser | None = None) -> None:
        self.max_file_size = max_file_size
        self.web_parser = web_parser

    def parse_file(self, filename: str, data: bytes) -> ParsedDocument:
        suffix = Path(filename).suffix.lower()
        if suffix not in self.ALLOWED_EXTENSIONS:
            raise ValidationError("仅支持 PDF、Word、Markdown 和 TXT 文件")
        if not data:
            raise ValidationError("文件为空，请重新选择")
        if len(data) > self.max_file_size:
            raise ValidationError("单个文件不能超过 20MB")
        if suffix == ".pdf":
            return PDFParser().parse(filename, data)
        if suffix in {".doc", ".docx"}:
            return WordParser().parse(filename, data)
        return TextParser().parse(filename, data)

    def parse_url(self, url: str) -> ParsedDocument:
        return (self.web_parser or WebParser()).parse(url)
