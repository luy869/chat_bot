import asyncio
from google import genai
from app.core.providers.base import LLMProvider, EmbeddingProvider


class GeminiLLMProvider(LLMProvider):
    """Google Gemini API プロバイダ"""

    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None):
        """
        model: 使用するGeminiモデル名
        api_key: Google API キー（環境変数 GEMINI_API_KEY から取得できる場合は省略可）
        """
        self.model = model
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

    async def generate(self, prompt: str) -> str:
        """テキスト生成（1回の応答）"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model, contents=prompt
            ),
        )
        return response.text if response else ""

    async def stream(self, prompt: str):
        """ストリーミング生成（複数の応答を逐次返す）"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model, contents=prompt, stream=True
            ),
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini Embedding API プロバイダ"""

    def __init__(self, model: str = "models/embedding-001", api_key: str | None = None):
        """
        model: 使用する埋め込みモデル名
        api_key: Google API キー
        """
        self.model = model
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

    async def embed(self, text: str) -> list[float]:
        """テキストをベクトルに変換"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.client.models.embed_content(
                model=self.model, contents=text
            ),
        )
        return result.embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """複数テキストをバッチ変換"""
        loop = asyncio.get_event_loop()
        embeddings = []
        for text in texts:
            result = await loop.run_in_executor(
                None,
                lambda t=text: self.client.models.embed_content(
                    model=self.model, contents=t
                ),
            )
            embeddings.append(result.embedding)
        return embeddings
