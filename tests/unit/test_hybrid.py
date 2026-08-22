import unittest

from rag_app.models import ChunkRecord, SearchHit
from rag_app.retrieval.hybrid import HybridRetriever


def result(key: str) -> SearchHit:
    return SearchHit(
        chunk=ChunkRecord.create(
            source=f"{key}.txt", doc_type="txt", content=key, index=0,
            location="body", chunk_profile="normal", content_hash=key,
        )
    )


class _VectorStore:
    last_top_k = None

    def search(self, query, top_k=10):
        self.last_top_k = top_k
        return [result("shared"), result("vector")]


class _BM25Store:
    last_top_k = None

    def search(self, query, top_k=10):
        self.last_top_k = top_k
        return [result("shared"), result("keyword")]


class HybridTests(unittest.TestCase):
    def test_search_fuses_and_deduplicates_two_retrievers(self):
        results = HybridRetriever(_VectorStore(), _BM25Store()).search("query")
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].chunk.content_hash, "shared")

    def test_vector_and_bm25_top_k_are_independently_configurable(self):
        vector = _VectorStore()
        bm25 = _BM25Store()
        HybridRetriever(vector, bm25, vector_top_k=7, bm25_top_k=4).search("query")
        self.assertEqual(vector.last_top_k, 7)
        self.assertEqual(bm25.last_top_k, 4)


if __name__ == "__main__":
    unittest.main()
