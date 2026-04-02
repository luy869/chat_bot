import os
import ollama
from app.core.providers.base import LLMProvider, EmbeddingProvider


class OllamaLLMProvider(LLMProvider):
    """Ollama ローカルLLM プロバイダ"""

    def __init__(self, model: str = "qwen3.5:9b", base_url: str = None):
        self.model = model
        if base_url is None:
            base_url = os.getenv("OLLAMA_HOST", "http://ollama:11434")
        self.client = ollama.AsyncClient(host=base_url)

    async def generate(self, messages: list[dict]) -> str:
        """テキスト生成（1回の応答）"""
        response = await self.client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            think=False,  # Qwen thinking を無効化
        )
        return response.get("message", {}).get("content", "")

    async def stream(self, messages: list[dict]):
        """ストリーミング生成"""
        stream_response = await self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            think=False,  # Qwen thinking を無効化
        )
        async for chunk in stream_response:
            yield chunk.get("message", {}).get("content", "")


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama テキスト埋め込みプロバイダ"""

    def __init__(self, model: str = None, base_url: str = None):
        if model is None:
            model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.model = model
        if base_url is None:
            base_url = os.getenv("OLLAMA_HOST", "http://ollama:11434")
        self.client = ollama.AsyncClient(host=base_url)

    async def embed(self, text: str) -> list[float]:
        """テキストをベクトルに変換"""
        response = await self.client.embed(model=self.model, input=text)
        return response.get("embeddings", [[]])[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """複数テキストをバッチ変換"""
        response = await self.client.embed(model=self.model, input=texts)
        return response.get("embeddings", [])
