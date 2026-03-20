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
        # 1. 関連チャンクを検索
        chunks = await self.retriever.retrieve(question, collection_name)

        # 2. 検索結果からLLM回答を生成
        answer = await self.generator.generate(question, chunks)

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
        # 1. 関連チャンクを検索
        chunks = await self.retriever.retrieve(question, collection_name)

        # 2. ストリーミング生成開始
        full_answer = ""
        async for text_chunk in self.generator.stream_generate(question, chunks):
            full_answer += text_chunk
            yield {
                "type": "chunk",
                "content": text_chunk,
            }

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
