#!/usr/bin/env python3
"""Small CLI to run sample ingestion for the repository.

Usage:
    python scripts/ingest_sample.py <path_or_url>

This will detect file type (pdf/md/txt or http(s) URL), create chunks, and
attempt to upsert them into a local Chroma collection named 'documents'.
"""
import sys
import os
from pathlib import Path
from rag_chroma.ingest import (
    read_pdf_file,
    read_markdown_file,
    read_text_file,
    read_url,
    prepare_chunks_from_document,
    upsert_chunks_to_chroma,
)


def detect_and_read(source: str):
    if source.startswith("http://") or source.startswith("https://"):
        return read_url(source)
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(source)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return read_pdf_file(str(p))
    if suffix in [".md", ".mdx"]:
        return read_markdown_file(str(p))
    if suffix == ".txt":
        return read_text_file(str(p))
    # fallback to text
    return read_text_file(str(p))


def main(argv):
    if len(argv) < 2:
        print("Usage: ingest_sample.py <path_or_url>")
        return 2
    source = argv[1]
    doc = detect_and_read(source)
    chunks = prepare_chunks_from_document(doc)
    n = upsert_chunks_to_chroma(chunks)
    print(f"Ingested {len(chunks)} chunks ({n} upserted) for {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
