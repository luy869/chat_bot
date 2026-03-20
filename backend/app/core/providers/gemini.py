import google.generativeai as genai
from app.core.providers.base import LLMProvider, EmbeddingProvider


class GeminiLLMProvider(LLMProvider):
    """Google Gemini API プロバイダ"""

    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None):
        """
        model: 使用するGeminiモデル名
        api_key: Google API キー（環境変数 GEMINI_API_KEY から取得できる場合は省略可）
        """
        if api_key:
            genai.configure(api_key=api_key)
        self.model = model
        self.client = genai.GenerativeModel(model)

    async def generate(self, prompt: str) -> str:
        """テキスト生成（1回の応答）"""
        response = self.client.generate_content(prompt)
        return response.text if response else ""

    async def stream(self, prompt: str):
        """ストリーミング生成（複数の応答を逐次返す）"""
        response = self.client.generate_content(prompt, stream=True)
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
        if api_key:
            genai.configure(api_key=api_key)
        self.model = model

    async def embed(self, text: str) -> list[float]:
        """テキストをベクトルに変換"""
        result = genai.embed_content(
            model=self.model,
            content=text,
        )
        return result["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """複数テキストをバッチ変換"""
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model,
                content=text,
            )
            embeddings.append(result["embedding"])
        return embeddings
