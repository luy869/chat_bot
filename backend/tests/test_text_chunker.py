import pytest
from app.core.ingestion.text import TextChunker


def test_simple_paragraphs():
    """空行区切りで段落に分割"""
    content = """First paragraph.

Second paragraph.

Third paragraph."""

    chunker = TextChunker()
    chunks = chunker.chunk(content, "notes.txt", "doc123")

    assert len(chunks) == 3
    assert chunks[0].content == "First paragraph."
    assert chunks[1].content == "Second paragraph."
    assert chunks[2].content == "Third paragraph."


def test_chunk_index_increments():
    """チャンクインデックスが増える"""
    content = "First.\n\nSecond.\n\nThird."
    chunker = TextChunker()
    chunks = chunker.chunk(content, "notes.txt", "doc123")

    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[2].chunk_index == 2


def test_long_paragraph_split():
    """max_charsを超える段落は分割される"""
    long_text = "word " * 300  # 1500文字超（max_chars=1000 を超える）
    chunker = TextChunker(max_chars=1000)
    chunks = chunker.chunk(long_text, "notes.txt", "doc123")

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.content) <= 1000


def test_heading_path_empty():
    """プレーンテキストは heading_path が空"""
    content = "Some text."
    chunker = TextChunker()
    chunks = chunker.chunk(content, "notes.txt", "doc123")

    assert chunks[0].heading_path == []


def test_page_number_none():
    """テキストファイルは page_number が None"""
    content = "Some text."
    chunker = TextChunker()
    chunks = chunker.chunk(content, "notes.txt", "doc123")

    assert chunks[0].page_number is None


def test_empty_content():
    """空のテキストはチャンクを生成しない"""
    chunker = TextChunker()
    chunks = chunker.chunk("", "notes.txt", "doc123")

    assert len(chunks) == 0


def test_document_metadata():
    """ドキュメントメタデータが正しく記録される"""
    content = "Some text."
    chunker = TextChunker()
    chunks = chunker.chunk(content, "readme.txt", "doc_xyz")

    assert chunks[0].source_file == "readme.txt"
    assert chunks[0].document_id == "doc_xyz"
