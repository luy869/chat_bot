from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from app.api.auth import require_api_key
from pydantic import BaseModel
from app.db.metadata import MetadataDB, Document
from app.core.vectorstore.chroma import ChromaVectorStore
from app.core.ingestion.markdown import MarkdownChunker
from app.core.ingestion.pdf import PDFChunker
from app.core.ingestion.text import TextChunker
import uuid

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentListResponse(BaseModel):
    """ドキュメント一覧レスポンス"""

    filename: str
    document_id: str
    file_size: int
    chunk_count: int
    created_at: str
    updated_at: str


async def get_metadata_db() -> MetadataDB:
    """メタデータDB の DI"""
    db = MetadataDB()
    await db.init()
    return db


async def get_vectorstore() -> ChromaVectorStore:
    """ベクトルストア DI（簡略版）"""
    # 実際にはプロバイダーとディレクトリをDIコンテナから取得
    from app.core.providers.ollama import OllamaEmbeddingProvider

    provider = OllamaEmbeddingProvider()
    return ChromaVectorStore(embedding_provider=provider)


@router.post("/upload", dependencies=[Depends(require_api_key)])
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Form("default"),
    metadata_db: MetadataDB = Depends(get_metadata_db),
    vectorstore: ChromaVectorStore = Depends(get_vectorstore),
):
    """
    ドキュメントをアップロードしてチャンキング & ベクトル化

    Args:
        file: アップロードファイル（.md, .pdf, .txt）
        collection_name: 保存先コレクション

    Returns:
        {"document_id": "...", "chunk_count": 5, "status": "success"}
    """
    # コレクション確認（なければ作成）
    await metadata_db.create_collection(collection_name)

    # ドキュメントID生成
    document_id = str(uuid.uuid4())

    # ファイル内容読み込み
    content = await file.read()
    filename = file.filename or "unknown"
    file_size = len(content)

    # ファイル形式に応じてチャンキング
    chunks = []
    if filename.endswith(".md"):
        chunker = MarkdownChunker()
        try:
            chunks = chunker.chunk(content.decode("utf-8"), filename, document_id)
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File encoding must be UTF-8")
    elif filename.endswith(".pdf"):
        chunker = PDFChunker()
        chunks = chunker.chunk(content, filename, document_id)
    elif filename.endswith(".txt"):
        chunker = TextChunker()
        try:
            chunks = chunker.chunk(content.decode("utf-8"), filename, document_id)
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File encoding must be UTF-8")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {filename}")

    # ベクトルストアに追加
    await vectorstore.add_chunks(chunks, collection_name)

    # メタデータDB に記録
    await metadata_db.add_document(
        document_id=document_id,
        collection_name=collection_name,
        filename=filename,
        file_size=file_size,
        chunk_count=len(chunks),
    )

    return {
        "status": "success",
        "document_id": document_id,
        "filename": filename,
        "chunk_count": len(chunks),
    }


@router.get("/{collection_name}")
async def list_documents(
    collection_name: str,
    metadata_db: MetadataDB = Depends(get_metadata_db),
):
    """コレクション内のドキュメント一覧"""
    docs = await metadata_db.get_documents(collection_name)
    return [
        DocumentListResponse(
            filename=doc.filename,
            document_id=doc.id,
            file_size=doc.file_size,
            chunk_count=doc.chunk_count,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )
        for doc in docs
    ]


@router.delete("/{document_id}", dependencies=[Depends(require_api_key)])
async def delete_document(
    document_id: str,
    collection_name: str,
    metadata_db: MetadataDB = Depends(get_metadata_db),
    vectorstore: ChromaVectorStore = Depends(get_vectorstore),
):
    """ドキュメント削除"""
    # ベクトルストアから document_id に紐づく全チャンクを削除
    await vectorstore.delete_document_chunks(document_id, collection_name)

    # メタデータDB から削除
    await metadata_db.delete_document(document_id)

    return {"status": "success", "document_id": document_id}
