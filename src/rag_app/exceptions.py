class RAGError(Exception):
    """Base class for expected application errors."""


class ValidationError(RAGError):
    """Input validation failed."""


class InvalidContentError(RAGError):
    """The source does not contain usable text."""


class DocumentParseError(RAGError):
    """A local document could not be parsed."""


class WebParseError(RAGError):
    """A web page could not be fetched or parsed."""


class RetrievalError(RAGError):
    """Retrieval or reranking failed."""


class LLMServiceError(RAGError):
    """The configured chat service failed."""
