"""Document loaders for PDF, Markdown, and Webpage sources."""

import os
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.ingestion.splitter import chunk_documents


def load_pdf(file_path: Union[str, Path]) -> list[Document]:
    """Load a PDF file and return chunked Documents.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of chunked Documents with enriched metadata.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    loader = PyPDFLoader(str(file_path))
    docs = loader.load()

    for i, doc in enumerate(docs):
        doc.metadata["source"] = str(file_path)
        doc.metadata["source_type"] = "pdf"
        doc.metadata["page"] = doc.metadata.get("page", i)

    return chunk_documents(docs)


def load_markdown(file_path: Union[str, Path]) -> list[Document]:
    """Load a Markdown file and return chunked Documents.

    Args:
        file_path: Path to the Markdown file.

    Returns:
        List of chunked Documents with enriched metadata.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    doc = Document(
        page_content=text,
        metadata={
            "source": str(file_path),
            "source_type": "markdown",
            "filename": file_path.name,
        },
    )
    return chunk_documents([doc])


def load_webpage(url: str, timeout: int = 30) -> list[Document]:
    """Load a webpage and return chunked Documents.

    Uses requests + BeautifulSoup to extract clean text from HTML.

    Args:
        url: The URL to fetch and parse.
        timeout: HTTP request timeout in seconds.

    Returns:
        List of chunked Documents with enriched metadata.
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {url}")

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    text_parts = []
    for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "blockquote"]):
        text = tag.get_text(strip=True)
        if text:
            text_parts.append(text)

    full_text = "\n\n".join(text_parts)

    doc = Document(
        page_content=full_text,
        metadata={
            "source": url,
            "source_type": "webpage",
            "title": soup.title.string if soup.title else "",
        },
    )
    return chunk_documents([doc])


def load_document(
    source: Union[str, Path],
) -> list[Document]:
    """Auto-detect document type and load accordingly.

    Args:
        source: File path (pdf/md) or URL.

    Returns:
        List of chunked Documents.
    """
    source_str = str(source)

    if source_str.startswith(("http://", "https://")):
        return load_webpage(source_str)

    ext = Path(source_str).suffix.lower()
    if ext == ".pdf":
        return load_pdf(source_str)
    elif ext in (".md", ".markdown", ".txt"):
        return load_markdown(source_str)
    else:
        raise ValueError(f"Unsupported document format: {ext}")
