import unittest

from rag_app.indexing.bm25_store import BM25Store
from rag_app.models import ChunkRecord


def chunk(name: str, content: str) -> ChunkRecord:
    return ChunkRecord.create(
        source=name,
        doc_type="txt",
        content=content,
        index=0,
        location="body",
        chunk_profile="normal",
    )


class _BM25:
    def __init__(self, corpus):
        self.corpus = corpus

    def get_scores(self, query):
        return [sum(tokens.count(token) for token in query) for tokens in self.corpus]


class BM25Tests(unittest.TestCase):
    def test_search_orders_keyword_match_first(self):
        store = BM25Store(tokenizer=lambda text: text.split(), bm25_factory=_BM25)
        store.rebuild([chunk("a", "年假 申请 流程"), chunk("b", "报销 申请 流程")])
        results = store.search("年假 申请", top_k=2)
        self.assertEqual(results[0].chunk.source, "a")
        self.assertGreater(results[0].bm25_score, results[1].bm25_score)

    def test_empty_index_returns_no_results(self):
        store = BM25Store(tokenizer=lambda text: text.split(), bm25_factory=_BM25)
        self.assertEqual(store.search("问题"), [])


if __name__ == "__main__":
    unittest.main()
