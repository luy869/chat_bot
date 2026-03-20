import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.core.providers.ollama import OllamaLLMProvider, OllamaEmbeddingProvider
from app.core.providers.gemini import GeminiLLMProvider, GeminiEmbeddingProvider


# ============ OllamaLLMProvider テスト ============

@pytest.mark.asyncio
async def test_ollama_llm_generate():
    """OllamaLLMProvider.generate() が正しく応答を返す"""
    with patch("app.core.providers.ollama.ollama.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.generate = AsyncMock(return_value={"response": "Hello from Ollama"})

        provider = OllamaLLMProvider(model="qwen3.5:9b")
        result = await provider.generate("What is AI?")

        assert result == "Hello from Ollama"
        mock_client.generate.assert_called_once()


@pytest.mark.asyncio
async def test_ollama_llm_stream():
    """OllamaLLMProvider.stream() がストリーミング応答を返す"""
    async def mock_stream(*args, **kwargs):
        for item in [{"response": "Hello "}, {"response": "from "}, {"response": "Ollama"}]:
            yield item

    with patch("app.core.providers.ollama.ollama.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.generate = mock_stream

        provider = OllamaLLMProvider(model="qwen3.5:9b")
        chunks = []
        async for chunk in provider.stream("What is AI?"):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert "".join(chunks) == "Hello from Ollama"


@pytest.mark.asyncio
async def test_ollama_embedding():
    """OllamaEmbeddingProvider.embed() がベクトルを返す"""
    with patch("app.core.providers.ollama.ollama.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.embed = AsyncMock(return_value={"embeddings": [[0.1, 0.2, 0.3, 0.4]]})

        provider = OllamaEmbeddingProvider(model="nomic-embed-text")
        result = await provider.embed("Hello world")

        assert result == [0.1, 0.2, 0.3, 0.4]
        mock_client.embed.assert_called_once()


@pytest.mark.asyncio
async def test_ollama_embedding_batch():
    """OllamaEmbeddingProvider.embed_batch() が複数ベクトルを返す"""
    with patch("app.core.providers.ollama.ollama.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.embed = AsyncMock(return_value={
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        })

        provider = OllamaEmbeddingProvider(model="nomic-embed-text")
        result = await provider.embed_batch(["Hello", "World"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]


# ============ GeminiLLMProvider テスト ============

@pytest.mark.asyncio
async def test_gemini_llm_generate():
    """GeminiLLMProvider.generate() が正しく応答を返す"""
    with patch("app.core.providers.gemini.genai.Client") as mock_client_class:
        mock_response = MagicMock()
        mock_response.text = "Hello from Gemini"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = GeminiLLMProvider(model="gemini-2.0-flash")
        result = await provider.generate("What is AI?")

        assert result == "Hello from Gemini"
        mock_client.models.generate_content.assert_called_once_with(
            model="gemini-2.0-flash", contents="What is AI?"
        )


@pytest.mark.asyncio
async def test_gemini_llm_stream():
    """GeminiLLMProvider.stream() がストリーミング応答を返す"""
    with patch("app.core.providers.gemini.genai.Client") as mock_client_class:
        chunk1 = MagicMock()
        chunk1.text = "Hello "
        chunk2 = MagicMock()
        chunk2.text = "from "
        chunk3 = MagicMock()
        chunk3.text = "Gemini"

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = [chunk1, chunk2, chunk3]
        mock_client_class.return_value = mock_client

        provider = GeminiLLMProvider(model="gemini-2.0-flash")
        chunks = []
        async for chunk in provider.stream("What is AI?"):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert "".join(chunks) == "Hello from Gemini"
        mock_client.models.generate_content.assert_called_once_with(
            model="gemini-2.0-flash", contents="What is AI?", stream=True
        )


@pytest.mark.asyncio
async def test_gemini_embedding():
    """GeminiEmbeddingProvider.embed() がベクトルを返す"""
    with patch("app.core.providers.gemini.genai.Client") as mock_client_class:
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4]
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = GeminiEmbeddingProvider(model="models/embedding-001")
        result = await provider.embed("Hello world")

        assert result == [0.1, 0.2, 0.3, 0.4]
        mock_client.models.embed_content.assert_called_once_with(
            model="models/embedding-001", contents="Hello world"
        )


@pytest.mark.asyncio
async def test_gemini_embedding_batch():
    """GeminiEmbeddingProvider.embed_batch() が複数ベクトルを返す"""
    with patch("app.core.providers.gemini.genai.Client") as mock_client_class:
        mock_response1 = MagicMock()
        mock_response1.embedding = [0.1, 0.2, 0.3]
        mock_response2 = MagicMock()
        mock_response2.embedding = [0.4, 0.5, 0.6]

        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = [mock_response1, mock_response2]
        mock_client_class.return_value = mock_client

        provider = GeminiEmbeddingProvider(model="models/embedding-001")
        result = await provider.embed_batch(["Hello", "World"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        assert mock_client.models.embed_content.call_count == 2
