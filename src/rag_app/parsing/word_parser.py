from __future__ import annotations

import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

from rag_app.exceptions import DocumentParseError
from rag_app.models import DocumentUnit, ParsedDocument
from rag_app.parsing.cleaners import clean_text


class WordParser:
    def parse(self, filename: str, data: bytes) -> ParsedDocument:
        suffix = Path(filename).suffix.lower()
        if suffix == ".doc":
            data = self._convert_legacy_doc(filename, data)
        try:
            from docx import Document

            document = Document(BytesIO(data))
            blocks: list[str] = []
            for block in document.iter_inner_content():
                if hasattr(block, "rows"):
                    for row in block.rows:
                        line = " | ".join(" ".join(cell.text.split()) for cell in row.cells)
                        if line.strip(" |"):
                            blocks.append(line)
                else:
                    text = block.text.strip()
                    if text:
                        blocks.append(text)
            cleaned = clean_text("\n".join(blocks))
        except Exception as exc:
            raise DocumentParseError("文档解析失败，请检查文件格式与完整性") from exc
        return ParsedDocument(
            source=filename,
            doc_type=suffix.lstrip("."),
            text=cleaned,
            units=[DocumentUnit(text=cleaned, location="body")],
        )

    @staticmethod
    def _convert_legacy_doc(filename: str, data: bytes) -> bytes:
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if not soffice:
            raise DocumentParseError("解析 .doc 需要安装 LibreOffice")
        with tempfile.TemporaryDirectory(prefix="rag-doc-") as directory:
            source = Path(directory, Path(filename).name)
            source.write_bytes(data)
            try:
                subprocess.run(
                    [soffice, "--headless", "--convert-to", "docx", "--outdir", directory, str(source)],
                    check=True,
                    capture_output=True,
                    timeout=60,
                )
                converted = source.with_suffix(".docx")
                if not converted.exists():
                    raise DocumentParseError(".doc 转换失败")
                return converted.read_bytes()
            except (subprocess.SubprocessError, OSError) as exc:
                raise DocumentParseError("文档解析失败，请检查文件格式与完整性") from exc
