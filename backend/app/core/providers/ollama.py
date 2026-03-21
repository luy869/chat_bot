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

    async def generate(self, prompt: str) -> str:
        """テキスト生成（1回の応答）"""
        response = await self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=False,
            think=False  # Qwen thinking を無効化
        )
        return response.get("response", "")

    async def stream(self, prompt: str):
        """ストリーミング生成"""
        stream_response = await self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=True,
            think=False  # Qwen thinking を無効化
        )
        async for chunk in stream_response:
            yield chunk.get("response", "")


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama テキスト埋め込みプロバイダ"""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = None):
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
