import unittest

from rag_app.models import ChunkRecord, SearchHit
from rag_app.ui.helpers import format_reference, profile_from_label


class UIHelperTests(unittest.TestCase):
    def test_auto_profile_returns_none(self):
        self.assertIsNone(profile_from_label("自动判断"))

    def test_technical_profile_is_mapped(self):
        self.assertEqual(profile_from_label("技术文档"), "technical")

    def test_reference_includes_source_location_and_score(self):
        item = SearchHit(chunk=ChunkRecord.create(
            source="制度.pdf", doc_type="pdf", content="正文", index=0,
            location="page:2", chunk_profile="normal",
        ), rerank_score=0.9123)
        self.assertEqual(format_reference(item), "制度.pdf · page:2 · 相关度 0.912")


if __name__ == "__main__":
    unittest.main()
