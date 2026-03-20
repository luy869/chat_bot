import ollama
from app.core.providers.base import LLMProvider, EmbeddingProvider


class OllamaLLMProvider(LLMProvider):
    """Ollama ローカルLLM プロバイダ"""

    def __init__(self, model: str = "gemma3:12b", base_url: str = "http://localhost:11434"):
        """
        model: 使用するOllamaモデル名（ollama pull gemma3:12b など）
        base_url: Ollamaサーバーのエンドポイント
        """
        self.model = model
        self.client = ollama.Client(host=base_url)

    async def generate(self, prompt: str) -> str:
        """テキスト生成（1回の応答）"""
        response = self.client.generate(model=self.model, prompt=prompt, stream=False)
        return response.get("response", "")

    async def stream(self, prompt: str):
        """ストリーミング生成（複数の応答を逐次返す）"""
        response = self.client.generate(model=self.model, prompt=prompt, stream=True)
        for chunk in response:
            yield chunk.get("response", "")


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama テキスト埋め込みプロバイダ"""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        """
        model: 使用する埋め込みモデル（ollama pull nomic-embed-text など）
        base_url: Ollamaサーバーのエンドポイント
        """
        self.model = model
        self.client = ollama.Client(host=base_url)

    async def embed(self, text: str) -> list[float]:
        """テキストをベクトルに変換"""
        response = self.client.embed(model=self.model, input=text)
        return response.get("embeddings", [[]])[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """複数テキストをバッチ変換"""
        response = self.client.embed(model=self.model, input=texts)
        return response.get("embeddings", [])
