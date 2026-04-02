from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """LLM（大規模言語モデル）の共通インターフェース"""

    @abstractmethod
    async def generate(self, messages: list[dict]) -> str:
        """テキスト生成（1回の応答）"""
        pass

    @abstractmethod
    async def stream(self, messages: list[dict]):
        """ストリーミング生成（複数の応答を逐次返す）"""
        pass


class EmbeddingProvider(ABC):
    """テキスト埋め込みの共通インターフェース"""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """テキストをベクトルに変換"""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """複数テキストをバッチ変換"""
        pass
