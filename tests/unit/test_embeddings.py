import numpy as np

from rag_app.indexing.chroma_store import BGEEmbeddings


class _EmbeddingModel:
    def __init__(self):
        self.inputs = []

    def encode(self, texts, **kwargs):
        self.inputs.append(list(texts))
        return np.asarray([[1.0, 0.0] for _ in texts])


def test_bge_query_has_instruction_but_documents_do_not():
    model = _EmbeddingModel()
    embeddings = BGEEmbeddings(model=model)
    embeddings.embed_documents(["文档正文"])
    embeddings.embed_query("年假有几天")
    assert model.inputs[0] == ["文档正文"]
    assert model.inputs[1] == ["为这个句子生成表示以用于检索相关文章：年假有几天"]
