import unittest

from rag_app.models import ChunkRecord, SearchHit
from rag_app.retrieval.rrf import reciprocal_rank_fusion


def hit(chunk_id: str, content_hash: str) -> SearchHit:
    chunk = ChunkRecord.create(
        source="source.txt",
        doc_type="txt",
        content=chunk_id,
        index=0,
        location="1",
        chunk_profile="normal",
        content_hash=content_hash,
    )
    return SearchHit(chunk=chunk)


class RRFTests(unittest.TestCase):
    def test_same_document_in_two_rankings_gets_combined_score(self):
        a = hit("a", "hash-a")
        b = hit("b", "hash-b")
        fused = reciprocal_rank_fusion([[a, b], [b, a]], k=60, limit=10)
        self.assertEqual({item.chunk.content_hash for item in fused}, {"hash-a", "hash-b"})
        self.assertAlmostEqual(fused[0].rrf_score, fused[1].rrf_score)

    def test_content_hash_deduplicates_chunks(self):
        first = hit("one", "same")
        duplicate = hit("two", "same")
        fused = reciprocal_rank_fusion([[first], [duplicate]], k=60, limit=10)
        self.assertEqual(len(fused), 1)


if __name__ == "__main__":
    unittest.main()
