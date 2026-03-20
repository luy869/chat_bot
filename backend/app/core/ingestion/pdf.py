from dataclasses import dataclass
from io import BytesIO
from app.models.chunk import Chunk


@dataclass
class PDFChunker:
    """PDF をページ単位でチャンキング。

    PDFには見出し構造が取りにくいため、ページ単位で分割する。
    """

    def _count_tokens(self, text: str) -> int:
        """簡易トークン数計算（1トークン ≈ 4文字）"""
        return max(1, len(text) // 4)

    def chunk(self, content: bytes, source_file: str, document_id: str) -> list[Chunk]:
        """
        PDFバイトデータをページ単位でチャンクに分割する。

        content: PDFファイルのバイトデータ
        source_file: "report.pdf"
        document_id: ドキュメントの一意ID
        """
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        chunks: list[Chunk] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()

            # テキストが空のページはスキップ（画像のみのページ等）
            if not text:
                continue

            chunks.append(Chunk(
                content=text,
                document_id=document_id,
                source_file=source_file,
                heading_path=[],         # PDFはページ構造のみ、見出し階層なし
                page_number=page_number,
                chunk_index=page_number - 1,
                token_count=self._count_tokens(text),
            ))

        return chunks
