import pytest
from app.core.ingestion.markdown import MarkdownChunker


def test_simple_markdown():
    """単純なMarkdownをチャンキング"""
    content = """# Title

Some text here.

## Section 1
More text.
"""
    chunker = MarkdownChunker()
    chunks = chunker.chunk(content, "test.md", "doc123")

    assert len(chunks) == 2
    assert chunks[0].content == "Some text here."
    assert chunks[0].heading_path == ["Title"]
    assert chunks[1].content == "More text."
    assert chunks[1].heading_path == ["Title", "Section 1"]


def test_nested_headings():
    """ネストされた見出し"""
    content = """# Chapter 1

Intro.

## Section 1.1

Text 1.1.

### Subsection 1.1.1

Text 1.1.1.
"""
    chunker = MarkdownChunker()
    chunks = chunker.chunk(content, "test.md", "doc123")

    assert len(chunks) == 3
    assert chunks[0].heading_path == ["Chapter 1"]
    assert chunks[1].heading_path == ["Chapter 1", "Section 1.1"]
    assert chunks[2].heading_path == ["Chapter 1", "Section 1.1", "Subsection 1.1.1"]


def test_heading_level_jump():
    """見出しレベルが飛ぶ場合（# → ### など）"""
    content = """# Title

Text.

### Jump to 3
Text 3.
"""
    chunker = MarkdownChunker()
    chunks = chunker.chunk(content, "test.md", "doc123")

    # level=3 の時: current_heading_path[:2] + ["Jump to 3"]
    # 直前は ["Title"] なので ["Title"] + ["Jump to 3"] = ["Title", "Jump to 3"]
    assert chunks[1].heading_path == ["Title", "Jump to 3"]


def test_empty_text_between_headings():
    """見出しの間にテキストがない場合、チャンクを生成しない"""
    content = """# Title

## Section 1

## Section 2

Text here.
"""
    chunker = MarkdownChunker()
    chunks = chunker.chunk(content, "test.md", "doc123")

    # Section 1 はテキストがないのでチャンク化されない
    assert len(chunks) == 1
    assert chunks[0].heading_path == ["Title", "Section 2"]
    assert chunks[0].content == "Text here."


def test_chunk_index_increments():
    """チャンクインデックスが順番に増える"""
    content = """# Title

Text 1.

## Section

Text 2.
"""
    chunker = MarkdownChunker()
    chunks = chunker.chunk(content, "test.md", "doc123")

    assert len(chunks) == 2
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_document_metadata():
    """ドキュメントメタデータが正しく記録される"""
    content = "# Title\n\nText."
    chunker = MarkdownChunker()
    chunks = chunker.chunk(content, "episodes.md", "doc_abc")

    assert chunks[0].source_file == "episodes.md"
    assert chunks[0].document_id == "doc_abc"
    assert chunks[0].page_number is None


def test_token_count_positive():
    """トークン数が正の値"""
    content = "# Title\n\nSome text."
    chunker = MarkdownChunker()
    chunks = chunker.chunk(content, "test.md", "doc123")

    assert chunks[0].token_count >= 1
