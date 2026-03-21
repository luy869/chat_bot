from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.db.metadata import MetadataDB, Collection
from app.core.vectorstore.chroma import ChromaVectorStore
from app.api.auth import require_api_key

router = APIRouter(prefix="/collections", tags=["collections"])


class CollectionResponse(BaseModel):
    """コレクションレスポンス"""

    name: str
    description: str
    document_count: int
    created_at: str
    updated_at: str


class SystemPromptRequest(BaseModel):
    """システムプロンプト設定リクエスト"""

    system_prompt: str


async def get_metadata_db() -> MetadataDB:
    """メタデータDB DI"""
    db = MetadataDB()
    await db.init()
    return db


async def get_vectorstore() -> ChromaVectorStore:
    """ベクトルストア DI"""
    from app.core.providers.ollama import OllamaEmbeddingProvider

    provider = OllamaEmbeddingProvider()
    return ChromaVectorStore(embedding_provider=provider)


@router.post("/{collection_name}", dependencies=[Depends(require_api_key)])
async def create_collection(
    collection_name: str,
    description: str = "",
    metadata_db: MetadataDB = Depends(get_metadata_db),
):
    """コレクション作成"""
    collection = await metadata_db.create_collection(collection_name, description)
    return CollectionResponse(
        name=collection.name,
        description=collection.description,
        document_count=collection.document_count,
        created_at=collection.created_at,
        updated_at=collection.updated_at,
    )


@router.get("/")
async def list_collections(
    metadata_db: MetadataDB = Depends(get_metadata_db),
):
    """全コレクション一覧"""
    collections = await metadata_db.get_collections()
    return [
        CollectionResponse(
            name=c.name,
            description=c.description,
            document_count=c.document_count,
            created_at=c.created_at,
            updated_at=c.updated_at,
        )
        for c in collections
    ]


@router.delete("/{collection_name}", dependencies=[Depends(require_api_key)])
async def delete_collection(
    collection_name: str,
    metadata_db: MetadataDB = Depends(get_metadata_db),
    vectorstore: ChromaVectorStore = Depends(get_vectorstore),
):
    """コレクション削除"""
    await vectorstore.delete_collection(collection_name)
    await metadata_db.delete_collection(collection_name)
    return {"status": "success", "collection_name": collection_name}


@router.put("/{collection_name}/system-prompt", dependencies=[Depends(require_api_key)])
async def set_system_prompt(
    collection_name: str,
    body: SystemPromptRequest,
    metadata_db: MetadataDB = Depends(get_metadata_db),
):
    """コレクションのシステムプロンプトを設定"""
    await metadata_db.set_system_prompt(collection_name, body.system_prompt)
    return {"status": "success", "collection_name": collection_name}
