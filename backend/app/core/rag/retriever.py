from dataclasses import dataclass
from app.models.chunk import Chunk
from app.core.vectorstore.chroma import ChromaVectorStore


@dataclass
class Retriever:
    """ベクトル検索で関連チャンクを取得"""

    vectorstore: ChromaVectorStore
    top_k: int = 5
    score_threshold: float = 0.7  # コサイン距離 0.7 以下のみ採用（類似度が高いものだけ）

    async def retrieve(self, query: str, collection_name: str) -> list[Chunk]:
        """クエリに関連するチャンクを検索"""
        return await self.vectorstore.search(
            query,
            collection_name,
            limit=self.top_k,
            score_threshold=self.score_threshold,
        )
