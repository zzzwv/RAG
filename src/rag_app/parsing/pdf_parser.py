from __future__ import annotations

from io import BytesIO
from typing import Any, Callable

from rag_app.exceptions import DocumentParseError, InvalidContentError
from rag_app.models import DocumentUnit, ParsedDocument
from rag_app.parsing.cleaners import clean_text, remove_repeated_page_lines


class PDFParser:
    def __init__(self, reader_factory: Callable[[Any], Any] | None = None) -> None:
        self.reader_factory = reader_factory

    def parse(self, filename: str, data: bytes) -> ParsedDocument:
        try:
            if self.reader_factory is None:
                from pypdf import PdfReader

                reader = PdfReader(BytesIO(data))
            else:
                reader = self.reader_factory(BytesIO(data))
            raw_pages = [(page.extract_text() or "") for page in reader.pages]
        except (InvalidContentError, DocumentParseError):
            raise
        except Exception as exc:
            raise DocumentParseError("文档解析失败，请检查文件格式与完整性") from exc
        pages = remove_repeated_page_lines(raw_pages)
        units: list[DocumentUnit] = []
        for number, page in enumerate(pages, start=1):
            try:
                content = clean_text(page)
            except InvalidContentError:
                continue
            units.append(DocumentUnit(text=content, location=f"page:{number}"))
        if not units:
            raise InvalidContentError("未提取到有效正文；扫描版 PDF 暂不支持 OCR")
        return ParsedDocument(
            source=filename,
            doc_type="pdf",
            text="\n\n".join(unit.text for unit in units),
            units=units,
            metadata={"page_count": len(raw_pages)},
        )
