import unittest

from rag_app.indexing.service import IngestionService
from rag_app.models import ChunkRecord, ParsedDocument


class _Parser:
    def parse_file(self, filename, data):
        return ParsedDocument(source=filename, doc_type="txt", text=data.decode())


class _Chunker:
    def split(self, document, profile=None):
        return [ChunkRecord.create(
            source=document.source, doc_type=document.doc_type, content=document.text,
            index=0, location="body", chunk_profile="normal",
        )]


class _Vector:
    def __init__(self):
        self.items = []

    def replace_source(self, source, chunks):
        self.items = list(chunks)

    def all_chunks(self):
        return self.items


class _BM25:
    def __init__(self):
        self.items = []

    def rebuild(self, chunks):
        self.items = list(chunks)


class IngestionTests(unittest.TestCase):
    def test_successful_ingestion_refreshes_bm25_from_vector_truth(self):
        vector = _Vector()
        bm25 = _BM25()
        service = IngestionService(_Parser(), _Chunker(), vector, bm25)
        result = service.ingest_file("policy.txt", b"policy content")
        self.assertEqual(result.chunk_count, 1)
        self.assertEqual(bm25.items, vector.items)


if __name__ == "__main__":
    unittest.main()
