from dataclasses import dataclass
from app.models.chunk import Chunk
from app.core.providers.base import LLMProvider


@dataclass
class Generator:
    """LLMで回答生成"""

    llm_provider: LLMProvider
    model: str = "gemma3:12b"
    temperature: float = 0.7
    max_tokens: int = 1000

    def _build_context(self, chunks: list[Chunk]) -> str:
        """検索結果からコンテキストを構築"""
        if not chunks:
            return ""

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            # 見出しがあれば含める
            if chunk.heading_path:
                heading = " > ".join(chunk.heading_path)
                context_parts.append(f"[{i}] {heading}\n{chunk.content}")
            else:
                # テキストやPDFの場合、ページ番号があれば含める
                if chunk.page_number:
                    context_parts.append(
                        f"[{i}] (Page {chunk.page_number})\n{chunk.content}"
                    )
                else:
                    context_parts.append(f"[{i}] {chunk.content}")

        return "\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """質問とコンテキストからプロンプトを構築"""
        system_prompt = """あなたは有用で正直なアシスタントです。
提供されたドキュメントの内容に基づいて、ユーザーの質問に答えてください。

以下のドキュメント内容を参考にしてください：

{context}

ドキュメント内に答えがない場合は、「提供されたドキュメント内には該当する情報がありません」と答えてください。"""

        return system_prompt.format(context=context) + f"\n\nユーザーの質問: {query}"

    async def generate(
        self, query: str, chunks: list[Chunk]
    ) -> str:
        """
        コンテキストを基に回答を生成（通常応答）

        Args:
            query: ユーザーの質問
            chunks: 検索結果のチャンク

        Returns:
            LLMが生成した回答テキスト
        """
        context = self._build_context(chunks)
        prompt = self._build_prompt(query, context)

        response = await self.llm_provider.generate(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response

    async def stream_generate(
        self, query: str, chunks: list[Chunk]
    ):
        """
        コンテキストを基に回答を生成（ストリーミング応答）

        Args:
            query: ユーザーの質問
            chunks: 検索結果のチャンク

        Yields:
            テキストの断片（ストリーミング）
        """
        context = self._build_context(chunks)
        prompt = self._build_prompt(query, context)

        async for chunk in self.llm_provider.stream(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ):
            yield chunk
