import time
from dataclasses import dataclass
from app.core.rag.retriever import Retriever
from app.core.rag.generator import Generator
from app.models.chunk import Chunk


@dataclass
class RAGResponse:
    """RAGパイプラインの応答"""

    answer: str
    source_chunks: list[Chunk]


@dataclass
class RAGPipeline:
    """RAGパイプライン：質問 → 検索 → 生成"""

    retriever: Retriever
    generator: Generator

    async def query(
        self, question: str, collection_name: str
    ) -> RAGResponse:
        """
        質問に対して RAG回答を生成

        Args:
            question: ユーザーの質問
            collection_name: 検索対象のコレクション

        Returns:
            RAGResponse（回答テキスト + 参照チャンク）
        """
        start_time = time.time()

        # 1. 関連チャンクを検索
        retrieval_start = time.time()
        chunks = await self.retriever.retrieve(question, collection_name)
        retrieval_time = time.time() - retrieval_start
        print(f"検索時間: {retrieval_time:.2f}秒")

        # 2. 検索結果からLLM回答を生成
        generation_start = time.time()
        answer = await self.generator.generate(question, chunks)
        generation_time = time.time() - generation_start
        print(f"生成時間: {generation_time:.2f}秒")

        total_time = time.time() - start_time
        print(f"合計時間: {total_time:.2f}秒")

        return RAGResponse(answer=answer, source_chunks=chunks)

    async def stream_query(self, question: str, collection_name: str):
        """
        質問に対してストリーミング回答を生成

        Args:
            question: ユーザーの質問
            collection_name: 検索対象のコレクション

        Yields:
            ストリーミングレスポンス（dict）
            - "type": "chunk" | "complete"
            - "content": テキスト断片 or 完全な RAGResponse
        """
        start_time = time.time()

        # 1. 関連チャンクを検索
        retrieval_start = time.time()
        chunks = await self.retriever.retrieve(question, collection_name)
        retrieval_time = time.time() - retrieval_start
        print(f"検索時間: {retrieval_time:.2f}秒")

        # 2. ストリーミング生成開始
        generation_start = time.time()
        full_answer = ""
        async for text_chunk in self.generator.stream_generate(question, chunks):
            full_answer += text_chunk
            yield {
                "type": "chunk",
                "content": text_chunk,
            }
        generation_time = time.time() - generation_start
        print(f"生成時間: {generation_time:.2f}秒")

        total_time = time.time() - start_time
        print(f"合計時間: {total_time:.2f}秒")

        # 3. ストリーム完了時に参照チャンクを返す
        yield {
            "type": "complete",
            "content": {
                "answer": full_answer,
                "source_chunks": [
                    {
                        "content": c.content,
                        "source_file": c.source_file,
                        "heading_path": c.heading_path,
                        "page_number": c.page_number,
                    }
                    for c in chunks
                ],
            },
        }
