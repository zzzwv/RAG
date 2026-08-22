import unittest

from rag_app.models import ChunkRecord


class ModelTests(unittest.TestCase):
    def test_chunk_id_is_deterministic(self):
        kwargs = dict(
            source="制度.txt",
            doc_type="txt",
            content="正文",
            index=3,
            location="section-1",
            chunk_profile="normal",
            content_hash="abc",
        )
        self.assertEqual(ChunkRecord.create(**kwargs).chunk_id, ChunkRecord.create(**kwargs).chunk_id)


if __name__ == "__main__":
    unittest.main()
