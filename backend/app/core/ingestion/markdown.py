from dataclasses import dataclass, field
from app.models.chunk import Chunk


@dataclass
class MarkdownChunker:
    """Markdown を見出し単位でチャンキング"""

    def _count_tokens(self, text: str) -> int:
        """
        文字列の長さに基づく簡易的なトークン数計算。
        1トークンを約4文字と仮定。
        """
        return max(1, len(text) // 4)

    def chunk(self, content: str, source_file: str, document_id: str) -> list[Chunk]:
        """
        content: Markdownテキスト全体
        source_file: "episodes.md"
        document_id: ドキュメントの一意ID

        返り値: Chunk のリスト
        """
        chunks = []
        lines = content.split('\n')

        current_heading_path = []  # 現在の見出し階層
        current_text = []          # 現在のテキスト
        chunk_index = 0

        for line in lines:
            if line.startswith('#'):
                # 見出しが来たら、前のチャンク確定
                if current_text:
                    text_content = '\n'.join(current_text).strip()
                    if text_content:
                        chunks.append(Chunk(
                            content=text_content,
                            document_id=document_id,
                            source_file=source_file,
                            heading_path=current_heading_path.copy(),
                            page_number=None,
                            chunk_index=chunk_index,
                            token_count=self._count_tokens(text_content)
                        ))
                        chunk_index += 1
                    current_text = []

                # 見出しの判定 → heading_path 更新
                level = len(line) - len(line.lstrip('#'))
                heading_text = line.lstrip('#').strip()
                if level > 0:
                    current_heading_path = current_heading_path[:level - 1] + [heading_text]
            
            # テキストの追加
            current_text.append(line)

        # 最後のチャンクを追加
        if current_text:
            text_content = '\n'.join(current_text).strip()
            if text_content:
                chunks.append(Chunk(
                    content=text_content,
                    document_id=document_id,
                    source_file=source_file,
                    heading_path=current_heading_path.copy(),
                    page_number=None,
                    chunk_index=chunk_index,
                    token_count=self._count_tokens(text_content)
                ))

        return chunks