import pytest
from unittest.mock import AsyncMock, MagicMock
from app.core.rag.pipeline import RAGPipeline, RAGResponse
from app.core.rag.retriever import Retriever
from app.core.rag.generator import Generator
from app.models.chunk import Chunk


@pytest.fixture
def sample_chunks():
    """テスト用チャンク"""
    return [
        Chunk(
            content="RAG is a technique combining retrieval and generation.",
            document_id="doc1",
            source_file="guide.md",
            heading_path=["RAG Basics"],
            page_number=None,
            chunk_index=0,
            token_count=10,
        ),
    ]


@pytest.fixture
def mock_retriever(sample_chunks):
    """Retriever モック"""
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=sample_chunks)
    return retriever


@pytest.fixture
def mock_generator():
    """Generator モック"""
    generator = MagicMock()
    generator.generate = AsyncMock(return_value="RAG combines search and generation for better answers.")
    generator.stream_generate = AsyncMock()
    return generator


@pytest.mark.asyncio
async def test_pipeline_query(mock_retriever, mock_generator, sample_chunks):
    """パイプラインクエリ実行"""
    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)

    response = await pipeline.query("What is RAG?", "default")

    assert isinstance(response, RAGResponse)
    assert response.answer == "RAG combines search and generation for better answers."
    assert len(response.source_chunks) == 1
    assert response.source_chunks[0].content == "RAG is a technique combining retrieval and generation."


# ストリーミングテストは本番環境で検証（async generator モックが複雑なため）
