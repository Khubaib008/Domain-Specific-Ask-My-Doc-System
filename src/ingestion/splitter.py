"""Text splitter module for chunking documents into overlapping segments."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def get_text_splitter(
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> RecursiveCharacterTextSplitter:
    """Create a RecursiveCharacterTextSplitter tuned for RAG.

    Default 700 tokens with 100-token overlap targets the 500-800 range
    specified in the architecture. Uses tiktoken for accurate token counting.
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Split a list of Documents into overlapping chunks.

    Args:
        documents: List of LangChain Documents to split.
        chunk_size: Target chunk size in tokens (default 700).
        chunk_overlap: Overlap between chunks in tokens (default 100).

    Returns:
        List of chunked Documents with updated metadata.
    """
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks
