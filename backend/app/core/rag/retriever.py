from dataclasses import dataclass
from app.models.chunk import Chunk
from app.core.vectorstore.chroma import ChromaVectorStore


@dataclass
class Retriever:
    """ベクトル検索で関連チャンクを取得"""

    vectorstore: ChromaVectorStore
    top_k: int = 5
    score_threshold: float | None = None  # None=フィルタなし、L2距離で指定する場合は ~1.5 以下が適切

    async def retrieve(self, query: str, collection_name: str) -> list[Chunk]:
        """クエリに関連するチャンクを検索"""
        return await self.vectorstore.search(
            query,
            collection_name,
            limit=self.top_k,
            score_threshold=self.score_threshold,
        )
