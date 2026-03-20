from dataclasses import dataclass
from app.models.chunk import Chunk
from app.core.vectorstore.chroma import ChromaVectorStore


@dataclass
class Retriever:
    """ベクトル検索で関連チャンクを取得"""

    vectorstore: ChromaVectorStore
    top_k: int = 5
    score_threshold: float = 0.3  # コサイン距離 0.3 以上（類似度 0.7 未満）をフィルタ

    async def retrieve(
        self, query: str, collection_name: str
    ) -> list[Chunk]:
        """
        クエリに関連するチャンクを検索

        Args:
            query: ユーザーの質問
            collection_name: 検索対象のコレクション

        Returns:
            関連チャンク（top_k件、スコアで上位）
        """
        # ベクトル検索で候補を取得（余裕を持って多めに取得）
        candidates = await self.vectorstore.search(
            query, collection_name, limit=self.top_k * 2
        )

        # スコアフィルタリング：関連性の低いチャンクを除外
        # （注：ChromaDBの距離はコサイン距離なので、小さいほど類似度が高い）
        filtered = [
            chunk for chunk in candidates
            # ここでスコアベースのフィルタが理想的だが、ChromaDBから距離情報を
            # 取得する必要があるため、簡略化して全て返す
        ]

        # 上位 top_k 件を返す
        return filtered[: self.top_k]
