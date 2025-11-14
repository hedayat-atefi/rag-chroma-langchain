"""Simple ingestion utilities for rag-chroma.

Provides loaders for text, markdown, and PDF files, a chunker that uses
simple token counting (tiktoken-compatible), fingerprinting, and a
helper to upsert into a Chroma collection.

This file aims to be PR-ready: minimal external coupling, clear config,
and testable functions.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from chromadb.errors import NoSuchCollectionError

import pdfplumber
import frontmatter
from bs4 import BeautifulSoup
import requests

# token counting; tiktoken import is optional - fall back to simple heuristic
try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency in tests
    tiktoken = None

logger = logging.getLogger(__name__)


@dataclass
class Document:
    id: str
    source_uri: str
    content: str
    content_type: str
    metadata: dict


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text_file(path: str, encoding: str = "utf-8") -> Document:
    with open(path, "r", encoding=encoding, errors="replace") as f:
        content = f.read()
    doc_id = sha256_text(content + "::" + os.path.abspath(path))
    metadata = {
        "source_path": os.path.abspath(path),
        "fetch_timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return Document(id=doc_id, source_uri=path, content=content, content_type="text", metadata=metadata)


def read_markdown_file(path: str, encoding: str = "utf-8") -> Document:
    with open(path, "r", encoding=encoding, errors="replace") as f:
        raw = f.read()
    fm = frontmatter.loads(raw)
    content = fm.content
    metadata = fm.metadata or {}
    metadata.update({
        "source_path": os.path.abspath(path),
        "fetch_timestamp": datetime.utcnow().isoformat() + "Z",
    })
    doc_id = sha256_text(content + json.dumps(metadata, sort_keys=True))
    return Document(id=doc_id, source_uri=path, content=content, content_type="markdown", metadata=metadata)


def read_pdf_file(path: str) -> Document:
    text_parts: List[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
    except Exception as e:
        logger.exception("Failed to read PDF %s: %s", path, e)
        raise

    content = "\n\n".join(text_parts)
    metadata = {
        "source_path": os.path.abspath(path),
        "page_count": len(text_parts),
        "fetch_timestamp": datetime.utcnow().isoformat() + "Z",
    }
    doc_id = sha256_text(content + metadata["source_path"]) if content else sha256_text(metadata["source_path"])
    return Document(id=doc_id, source_uri=path, content=content, content_type="pdf", metadata=metadata)


def read_url(url: str, timeout: int = 10) -> Document:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    html = resp.text
    # Prefer the faster/stricter 'lxml' parser when available; fall back to
    # Python's built-in 'html.parser' if it's not installed in the environment.
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    # Try to extract main textual content (naive)
    main = soup.find("main") or soup.find("article") or soup.body
    content = main.get_text(separator="\n") if main else soup.get_text(separator="\n")
    metadata = {
        "source_url": url,
        "status_code": resp.status_code,
        "content_type": resp.headers.get("content-type"),
        "fetch_timestamp": datetime.utcnow().isoformat() + "Z",
    }
    doc_id = sha256_text(content + url)
    return Document(id=doc_id, source_uri=url, content=content, content_type="html", metadata=metadata)


# Tokenization / chunking helpers
def count_tokens(text: str, model: Optional[str] = None) -> int:
    if tiktoken is not None and model is not None:
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    # fallback heuristic: 1 token per 4 characters
    return max(1, len(text) // 4)


def chunk_text(text: str, chunk_tokens: int = 500, chunk_overlap: int = 50, model: Optional[str] = None) -> List[Tuple[str, int, int]]:
    """Return list of (chunk_text, char_start, char_end).

    This is a conservative chunker: it splits on paragraph boundaries where possible,
    otherwise it falls back to token-length sliding window. Character indices are returned
    for provenance.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[Tuple[str, int, int]] = []

    char_cursor = 0
    for p in paragraphs:
        p_start = text.find(p, char_cursor)
        if p_start == -1:
            p_start = char_cursor
        p_end = p_start + len(p)

        # if paragraph small, try to append to previous chunk
        tokens = count_tokens(p, model=model)
        if tokens <= chunk_tokens:
            # either start a new chunk or append to last if it doesn't exceed
            if not chunks:
                chunks.append((p, p_start, p_end))
            else:
                last_text, last_start, last_end = chunks[-1]
                last_tokens = count_tokens(last_text, model=model)
                if last_tokens + tokens <= chunk_tokens:
                    new_text = last_text + "\n\n" + p
                    chunks[-1] = (new_text, last_start, p_end)
                else:
                    chunks.append((p, p_start, p_end))
        else:
            # paragraph itself large; do sliding window by characters approximating tokens
            approx_chars_per_token = 4
            window_chars = chunk_tokens * approx_chars_per_token
            overlap_chars = chunk_overlap * approx_chars_per_token
            start = p_start
            while start < p_end:
                end = min(p_end, start + window_chars)
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append((chunk, start, end))
                start = end - overlap_chars
        char_cursor = p_end

    # Final pass: if overlap requested, we keep as-is because tokens-based overlap is approximated
    return chunks


def prepare_chunks_from_document(doc: Document, chunk_tokens: int = 500, chunk_overlap: int = 50, model: Optional[str] = None) -> List[dict]:
    chunks = chunk_text(doc.content, chunk_tokens=chunk_tokens, chunk_overlap=chunk_overlap, model=model)
    out = []
    for idx, (text, start, end) in enumerate(chunks):
        chunk_id = sha256_text(doc.id + f"::chunk::{idx}")
        item = {
            "id": chunk_id,
            "document_id": doc.id,
            "text": text,
            "metadata": {
                **doc.metadata,
                "parent_id": doc.id,
                "chunk_index": idx,
                "char_start": start,
                "char_end": end,
                "token_count": count_tokens(text, model=model),
            },
        }
        out.append(item)
    return out


# Chroma upsert helper
DEFAULT_CHROMA_SETTINGS = Settings()


def get_chroma_client(chroma_server: Optional[str] = None):
    if chroma_server:
        return chromadb.Client(Settings(chroma_server=chroma_server))
    return chromadb.Client(DEFAULT_CHROMA_SETTINGS)


def upsert_chunks_to_chroma(chunks: List[dict], collection_name: str = "documents", chroma_server: Optional[str] = None):
    client = get_chroma_client(chroma_server)
    try:
        col = client.get_collection(collection_name)
    except NoSuchCollectionError:
        col = client.create_collection(collection_name)

    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    documents = [c["text"] for c in chunks]

    # embeddings are expected to be created by an embeddings provider; Chroma can accept raw text
    # but usually you pass embeddings. Here we'll use Chroma's internal embedding creation if configured,
    # otherwise store texts as "documents" and let external pipeline embed them later.
    try:
        col.add(ids=ids, metadatas=metadatas, documents=documents)
    except Exception:
        # try upsert (safe for duplicates)
        col.upsert(ids=ids, metadatas=metadatas, documents=documents)

    return len(ids)


if __name__ == "__main__":
    print("rag_chroma.ingest module - import and use functions programmatically")
