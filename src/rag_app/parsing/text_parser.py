from __future__ import annotations

from pathlib import Path

from rag_app.models import DocumentUnit, ParsedDocument
from rag_app.parsing.cleaners import clean_text


class TextParser:
    ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "big5")

    @staticmethod
    def _decode(data: bytes) -> tuple[str, str]:
        for encoding in TextParser.ENCODINGS:
            try:
                return data.decode(encoding), encoding
            except UnicodeDecodeError:
                continue
        try:
            from charset_normalizer import from_bytes

            match = from_bytes(data).best()
            if match is not None and match.encoding and match.percent_chaos <= 20:
                return str(match), match.encoding
        except ImportError:
            pass
        return data.decode("latin-1"), "latin-1"

    def parse(self, filename: str, data: bytes) -> ParsedDocument:
        text, encoding = self._decode(data)
        cleaned = clean_text(text)
        doc_type = Path(filename).suffix.lower().lstrip(".")
        return ParsedDocument(
            source=filename,
            doc_type=doc_type,
            text=cleaned,
            units=[DocumentUnit(text=cleaned, location="body")],
            metadata={"encoding": encoding},
        )
