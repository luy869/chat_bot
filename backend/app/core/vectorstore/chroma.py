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

        # コレクションを取得 or 作成（コサイン距離を使用）
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

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
        query_embedding = await self.embedding_provider.embed(query)

        # "all" の場合は全コレクション検索
        if collection_name == "all":
            return await self._search_all_collections(
                query_embedding, limit, score_threshold
            )

        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )

        return self._parse_search_results(results, score_threshold)

    async def _search_all_collections(
        self, query_embedding: list[float], limit: int, score_threshold: float | None
    ) -> list[Chunk]:
        """全コレクションから検索"""
        all_chunks = []

        # すべてのコレクション取得
        collections = self.client.list_collections()

        for collection_obj in collections:
            collection = self.client.get_collection(name=collection_obj.name)

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit * 2,  # 多めに取得（マージ後に絞る）
                include=["documents", "metadatas", "distances"],
            )

            chunks = self._parse_search_results(results, score_threshold)
            all_chunks.extend(chunks)

        # 距離でソートして重複排除（同じ content は一つだけ）
        seen = set()
        sorted_chunks = sorted(
            all_chunks,
            key=lambda c: (
                self._get_distance_from_chunk(c),
                c.source_file,
                c.chunk_index,
            ),
        )

        unique_chunks = []
        for chunk in sorted_chunks:
            chunk_key = (chunk.content, chunk.source_file)
            if chunk_key not in seen:
                seen.add(chunk_key)
                unique_chunks.append(chunk)
                if len(unique_chunks) >= limit:
                    break

        return unique_chunks

    def _parse_search_results(
        self, results: dict, score_threshold: float | None
    ) -> list[Chunk]:
        """ChromaDB 検索結果をチャンク化"""
        chunks = []
        if not results["ids"] or len(results["ids"]) == 0:
            return chunks

        for i, chunk_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]

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
            # 距離情報を一時的に保存（ソート用）
            chunk._distance = distance
            chunks.append(chunk)

        return chunks

    def _get_distance_from_chunk(self, chunk: Chunk) -> float:
        """チャンクの距離情報を取得（ソート用）"""
        return getattr(chunk, "_distance", 2.0)  # デフォルト 2.0（最大距離）

    async def update_chunk(self, chunk: Chunk, collection_name: str) -> None:
        """既存チャンクを更新"""
        chunk_id = f"{chunk.document_id}:{chunk.chunk_index}"
        await self.delete_chunk(chunk_id, collection_name)
        await self.add_chunks([chunk], collection_name)

    async def delete_document_chunks(self, document_id: str, collection_name: str) -> None:
        """document_id に紐づく全チャンクを削除"""
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        collection.delete(where={"document_id": document_id})

    async def delete_chunk(self, chunk_id: str, collection_name: str) -> None:
        """チャンクを削除"""
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        collection.delete(ids=[chunk_id])

    async def delete_collection(self, collection_name: str) -> None:
        """コレクション全体を削除"""
        self.client.delete_collection(name=collection_name)