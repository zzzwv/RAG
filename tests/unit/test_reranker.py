import unittest

from rag_app.models import ChunkRecord, SearchHit
from rag_app.retrieval.reranker import BGEReranker


def hit(key: str) -> SearchHit:
    return SearchHit(chunk=ChunkRecord.create(
        source=f"{key}.txt", doc_type="txt", content=key, index=0,
        location="body", chunk_profile="normal", content_hash=key,
    ))


class _Model:
    def __init__(self):
        self.kwargs = {}

    def predict(self, pairs, **kwargs):
        self.kwargs = kwargs
        scores = {"a": -2.0, "b": 2.0, "c": 1.0, "d": 0.5}
        return [scores[document] for _, document in pairs]


class RerankerTests(unittest.TestCase):
    def test_threshold_filters_and_top_n_limits_results(self):
        model = _Model()
        reranker = BGEReranker(model=model, threshold=0.6, top_n=2)
        results = reranker.rerank("question", [hit("a"), hit("b"), hit("c"), hit("d")])
        self.assertEqual([item.chunk.content for item in results], ["b", "c"])
        self.assertTrue(all(item.rerank_score >= 0.6 for item in results))
        self.assertEqual([item.rank for item in results], [1, 2])
        self.assertIn("activation_fn", model.kwargs)


if __name__ == "__main__":
    unittest.main()
