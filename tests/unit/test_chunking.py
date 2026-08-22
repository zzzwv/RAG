import unittest

from rag_app.chunking.classifier import ChunkProfile
from rag_app.chunking.service import ChunkingService
from rag_app.models import DocumentUnit, ParsedDocument


class ChunkingTests(unittest.TestCase):
    def test_normal_profile_uses_configured_size_and_overlap(self):
        service = ChunkingService(normal_size=20, normal_overlap=5, technical_size=10, technical_overlap=2)
        document = ParsedDocument(
            source="policy.txt",
            doc_type="txt",
            text="第一条规定。第二条规定。第三条规定。第四条规定。",
            units=[DocumentUnit(text="第一条规定。第二条规定。第三条规定。第四条规定。", location="body")],
        )
        chunks = service.split(document, profile=ChunkProfile.NORMAL)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunk.chunk_profile == "normal" for chunk in chunks))
        self.assertTrue(all(chunk.location == "body" for chunk in chunks))

    def test_empty_or_symbol_chunks_are_not_returned(self):
        service = ChunkingService(normal_size=20, normal_overlap=5)
        document = ParsedDocument(source="x.txt", doc_type="txt", text="---\n有效正文内容。")
        chunks = service.split(document, profile=ChunkProfile.NORMAL)
        self.assertEqual([chunk.content for chunk in chunks], ["有效正文内容。"])


if __name__ == "__main__":
    unittest.main()
