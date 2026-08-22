import unittest

from rag_app.exceptions import ValidationError
from rag_app.parsing.pdf_parser import PDFParser
from rag_app.parsing.pipeline import DocumentParsingPipeline
from rag_app.parsing.text_parser import TextParser


class _FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class _FakeReader:
    def __init__(self, _stream):
        self.pages = [
            _FakePage("企业手册\n第一页内容\n第 1 页"),
            _FakePage("企业手册\n第二页内容\n第 2 页"),
        ]


class ParsingTests(unittest.TestCase):
    def test_gb18030_text_is_decoded(self):
        parsed = TextParser().parse("制度.txt", "中文制度正文".encode("gb18030"))
        self.assertEqual(parsed.text, "中文制度正文")
        self.assertEqual(parsed.doc_type, "txt")

    def test_pdf_parser_keeps_page_locations_and_removes_headers(self):
        parsed = PDFParser(reader_factory=_FakeReader).parse("handbook.pdf", b"fake")
        self.assertEqual([unit.location for unit in parsed.units], ["page:1", "page:2"])
        self.assertNotIn("企业手册", parsed.text)
        self.assertIn("第二页内容", parsed.text)

    def test_pipeline_rejects_unsupported_extension(self):
        with self.assertRaises(ValidationError):
            DocumentParsingPipeline().parse_file("data.xlsx", b"content")

    def test_pipeline_rejects_file_over_limit(self):
        pipeline = DocumentParsingPipeline(max_file_size=4)
        with self.assertRaises(ValidationError):
            pipeline.parse_file("data.txt", b"12345")


if __name__ == "__main__":
    unittest.main()
