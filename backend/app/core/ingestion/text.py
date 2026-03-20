from dataclasses import dataclass
from app.models.chunk import Chunk


@dataclass
class TextChunker:
    """プレーンテキストを段落単位でチャンキング。

    空行区切りで段落に分割し、長すぎる段落はmax_chars単位で分割する。
    """

    max_chars: int = 1000  # 1チャンクの最大文字数

    def _count_tokens(self, text: str) -> int:
        """簡易トークン数計算（1トークン ≈ 4文字）"""
        return max(1, len(text) // 4)

    def chunk(self, content: str, source_file: str, document_id: str) -> list[Chunk]:
        """
        テキストを段落（空行区切り）でチャンクに分割する。
        段落がmax_charsを超える場合は、さらに分割する。
        """
        chunks: list[Chunk] = []
        chunk_index = 0

        # 空行区切りで段落に分割
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        for paragraph in paragraphs:
            # 段落がmax_chars以内ならそのままチャンク化
            if len(paragraph) <= self.max_chars:
                chunks.append(Chunk(
                    content=paragraph,
                    document_id=document_id,
                    source_file=source_file,
                    heading_path=[],
                    page_number=None,
                    chunk_index=chunk_index,
                    token_count=self._count_tokens(paragraph),
                ))
                chunk_index += 1
            else:
                # 長すぎる段落はmax_chars単位で分割
                for sub_chunk in self._split_long_paragraph(paragraph):
                    chunks.append(Chunk(
                        content=sub_chunk,
                        document_id=document_id,
                        source_file=source_file,
                        heading_path=[],
                        page_number=None,
                        chunk_index=chunk_index,
                        token_count=self._count_tokens(sub_chunk),
                    ))
                    chunk_index += 1

        return chunks

    def _split_long_paragraph(self, text: str) -> list[str]:
        """max_chars を超える段落をmax_chars単位で分割する"""
        parts = []
        while len(text) > self.max_chars:
            # max_chars位置から前に向かって最後のスペースを探す（単語の途中で切らない）
            split_at = text.rfind(" ", 0, self.max_chars)
            if split_at == -1:
                # スペースが見つからない場合は強制的にmax_chars位置で切る
                split_at = self.max_chars
            parts.append(text[:split_at].strip())
            text = text[split_at:].strip()
        if text:
            parts.append(text)
        return parts
