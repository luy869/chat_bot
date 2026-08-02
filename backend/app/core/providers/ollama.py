import os
import ollama
from app.core.providers.base import LLMProvider, EmbeddingProvider

# モデルをVRAMに保持する時間。未設定なら Ollama 既定（5分）に任せる。
# GPU は他サービスと共有しているため、長く保持させたい場合だけ明示的に設定する
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE") or None


class OllamaLLMProvider(LLMProvider):
    """Ollama ローカルLLM プロバイダ"""

    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or os.getenv("OLLAMA_LLM_MODEL", "qwen3.5:9b")
        if base_url is None:
            base_url = os.getenv("OLLAMA_HOST", "http://ollama:11434")
        self.client = ollama.AsyncClient(host=base_url)

    async def generate(self, messages: list[dict]) -> str:
        """テキスト生成（1回の応答）"""
        response = await self.client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            think=False,
            keep_alive=KEEP_ALIVE,
        )
        return response.get("message", {}).get("content", "")

    async def stream(self, messages: list[dict]):
        """ストリーミング生成"""
        stream_response = await self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            think=False,
            keep_alive=KEEP_ALIVE,
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
        response = await self.client.embed(
            model=self.model, input=text, keep_alive=KEEP_ALIVE
        )
        return response.get("embeddings", [[]])[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """複数テキストをバッチ変換"""
        response = await self.client.embed(
            model=self.model, input=texts, keep_alive=KEEP_ALIVE
        )
        return response.get("embeddings", [])
