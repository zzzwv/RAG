import tempfile

from rag_app.indexing.chroma_store import ChromaStore
from rag_app.models import ChunkRecord


class DeterministicEmbeddings:
    @staticmethod
    def _vector(text):
        return [float(len(text)), float(text.count("年假")), 1.0]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


def make_chunk(content, index=0):
    return ChunkRecord.create(
        source="policy.txt", doc_type="txt", content=content, index=index,
        location="body", chunk_profile="normal",
    )


def test_chroma_persists_replaces_and_clears_source():
    with tempfile.TemporaryDirectory() as directory:
        store = ChromaStore(directory, collection_name="test_collection", embedding_function=DeterministicEmbeddings())
        store.replace_source("policy.txt", [make_chunk("年假五天"), make_chunk("申请流程", 1)])
        assert store.count() == 2
        assert len(store.all_chunks()) == 2
        assert store.search("年假", top_k=1)[0].chunk.source == "policy.txt"

        store.replace_source("policy.txt", [make_chunk("年假六天")])
        assert store.count() == 1
        assert store.all_chunks()[0].content == "年假六天"

        reopened = ChromaStore(directory, collection_name="test_collection", embedding_function=DeterministicEmbeddings())
        assert reopened.count() == 1
        reopened.clear()
        assert reopened.count() == 0
        reopened.close()
        store.close()
