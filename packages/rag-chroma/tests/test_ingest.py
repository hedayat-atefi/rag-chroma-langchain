import os
import tempfile
import pytest
from rag_chroma.ingest import (
    read_text_file,
    read_markdown_file,
    chunk_text,
    prepare_chunks_from_document,
)


def test_read_text_and_chunking():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tf:
        tf.write("Hello world\n\nThis is a test document. " * 40)
        tf.flush()
        path = tf.name
    try:
        doc = read_text_file(path)
        assert doc.content
        chunks = prepare_chunks_from_document(doc, chunk_tokens=100, chunk_overlap=10)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
    finally:
        os.unlink(path)


def test_read_markdown_frontmatter_and_chunk():
    md = """---\ntitle: Test Doc\nauthor: QA\n---\n\n# Heading\n\nParagraph one.\n\nParagraph two."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", delete=False) as tf:
        tf.write(md)
        tf.flush()
        path = tf.name
    try:
        doc = read_markdown_file(path)
        assert doc.metadata.get("title") == "Test Doc"
        chunks = prepare_chunks_from_document(doc, chunk_tokens=50)
        assert chunks
    finally:
        os.unlink(path)

@pytest.mark.skipif(not os.path.exists("./tests/sample.pdf"), reason="no sample pdf")
def test_read_pdf():
    from rag_chroma.ingest import read_pdf_file
    doc = read_pdf_file("./tests/sample.pdf")
    assert doc.content is not None
