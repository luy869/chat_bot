from dataclasses import dataclass
from app.models.chunk import Chunk
from app.core.providers.base import LLMProvider


@dataclass
class Generator:
    """LLMで回答生成"""

    llm_provider: LLMProvider

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

    DEFAULT_SYSTEM_PROMPT = """あなたは有用で正直なアシスタントです。
提供されたドキュメントの内容に基づいて、ユーザーの質問に答えてください。
ドキュメント内に答えがない場合は、「提供されたドキュメント内には該当する情報がありません」と答えてください。"""

    def _build_prompt(self, query: str, context: str, system_prompt: str | None = None) -> str:
        """質問とコンテキストからプロンプトを構築"""
        base = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        return f"{base}\n\n以下のドキュメント内容を参考にしてください：\n\n{context}\n\nユーザーの質問: {query}"

    async def generate(
        self, query: str, chunks: list[Chunk], system_prompt: str | None = None
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
        prompt = self._build_prompt(query, context, system_prompt)

        response = await self.llm_provider.generate(prompt=prompt)
        return response

    async def stream_generate(
        self, query: str, chunks: list[Chunk], system_prompt: str | None = None
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
        prompt = self._build_prompt(query, context, system_prompt)

        async for chunk in self.llm_provider.stream(prompt=prompt):
            yield chunk
