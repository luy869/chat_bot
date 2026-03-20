from dataclasses import dataclass
import chromadb
from app.models.chunk import Chunk
from app.core.providers.base import EmbeddingProvider


@dataclass
class ChromaVectorStore:
    """ChromaDB ベクトルストア ラッパー"""

    embedding_provider: EmbeddingProvider
    persist_directory: str = "./chroma_data"

    def __post_init__(self):
        """初期化時に ChromaDB クライアント作成"""
        self.client = chromadb.PersistentClient(path=self.persist_directory)

    async def add_chunks(self, chunks: list[Chunk], collection_name: str) -> None:
        """チャンク群をコレクションに追加"""
        if not chunks:
            return

        # コレクションを取得 or 作成
        collection = self.client.get_or_create_collection(name=collection_name)

        # 1. chunk.content をベクトル化
        contents = [chunk.content for chunk in chunks]
        embeddings = await self.embedding_provider.embed_batch(contents)

        # 2. ID 生成
        ids = [f"{chunk.document_id}:{chunk.chunk_index}" for chunk in chunks]

        # 3. メタデータを辞書化（ChromaDB は単純な型のみ対応）
        metadatas = []
        for chunk in chunks:
            metadata = {
                "source_file": chunk.source_file,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number or -1,  # None を避ける
                "token_count": chunk.token_count,
                "heading_path": "|".join(chunk.heading_path) if chunk.heading_path else "",
            }
            metadatas.append(metadata)

        # 4. ChromaDB に追加
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )

    async def search(
        self, query: str, collection_name: str, limit: int = 5, score_threshold: float | None = None
    ) -> list[Chunk]:
        """クエリから関連チャンク検索"""
        collection = self.client.get_or_create_collection(name=collection_name)

        query_embedding = await self.embedding_provider.embed(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if results["ids"] and len(results["ids"]) > 0:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]

                # score_threshold はコサイン距離（小さいほど類似度が高い）
                if score_threshold is not None and distance > score_threshold:
                    continue

                metadata = results["metadatas"][0][i]
                content = results["documents"][0][i]

                heading_path_str = metadata.get("heading_path", "")
                heading_path = heading_path_str.split("|") if heading_path_str else []

                page_number = metadata.get("page_number", -1)
                page_number = None if page_number == -1 else page_number

                chunk = Chunk(
                    content=content,
                    document_id=metadata["document_id"],
                    source_file=metadata["source_file"],
                    heading_path=heading_path,
                    page_number=page_number,
                    chunk_index=metadata["chunk_index"],
                    token_count=metadata["token_count"],
                )
                chunks.append(chunk)

        return chunks

    async def update_chunk(self, chunk: Chunk, collection_name: str) -> None:
        """既存チャンクを更新"""
        chunk_id = f"{chunk.document_id}:{chunk.chunk_index}"
        await self.delete_chunk(chunk_id, collection_name)
        await self.add_chunks([chunk], collection_name)

    async def delete_document_chunks(self, document_id: str, collection_name: str) -> None:
        """document_id に紐づく全チャンクを削除"""
        collection = self.client.get_or_create_collection(name=collection_name)
        collection.delete(where={"document_id": document_id})

    async def delete_chunk(self, chunk_id: str, collection_name: str) -> None:
        """チャンクを削除"""
        collection = self.client.get_or_create_collection(name=collection_name)
        collection.delete(ids=[chunk_id])

    async def delete_collection(self, collection_name: str) -> None:
        """コレクション全体を削除"""
        self.client.delete_collection(name=collection_name)