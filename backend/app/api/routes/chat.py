import os
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from app.core.rag.pipeline import RAGPipeline, RAGResponse
from app.core.rag.retriever import Retriever
from app.core.rag.generator import Generator
from app.db.metadata import MetadataDB
from app.api.auth import require_api_key
from app.core.rate_limit import limiter, CHAT_RATE_LIMIT

router = APIRouter(prefix="/chat", tags=["chat"])

# このエンドポイントは公開ウィジェット（luy_web/widget）が直接叩く唯一のAPI。
# collection_name はクライアントが自由な文字列を送れるため、
# 許可リストで縛らないと "all" や非公開コレクション名を指定して横断検索されてしまう。
# X-API-Key の有無に関わらず必ず適用する（Cloudflare Worker は Origin ヘッダーが
# 一致してさえいれば任意のクライアントにも同じ API キーを自動付与してしまうため、
# 「キーを持っている=正規の呼び出し元」という前提が成立しない。詳細はDEVLOG.md参照）
PUBLIC_CHAT_COLLECTIONS = {
    name.strip()
    for name in os.getenv("PUBLIC_CHAT_COLLECTIONS", "default").split(",")
    if name.strip()
}


def _validate_collection_name(collection_name: str) -> None:
    """公開チャットエンドポイントで許可されたコレクションかを検証する"""
    if collection_name not in PUBLIC_CHAT_COLLECTIONS:
        raise HTTPException(
            status_code=403,
            detail="This collection is not accessible from the chat endpoint",
        )


class ChatRequest(BaseModel):
    """チャットリクエスト"""

    question: str
    collection_name: str = "default"
    stream: bool = False


class ChatResponse(BaseModel):
    """チャットレスポンス"""

    answer: str
    source_files: list[str]


async def get_rag_pipeline() -> RAGPipeline:
    """RAGパイプラインのDI"""
    from app.core.providers.ollama import OllamaEmbeddingProvider, OllamaLLMProvider
    from app.core.vectorstore.chroma import ChromaVectorStore

    embedding_provider = OllamaEmbeddingProvider()
    vectorstore = ChromaVectorStore(embedding_provider=embedding_provider)
    top_k = int(os.getenv("RAG_TOP_K", "10"))
    retriever = Retriever(vectorstore=vectorstore, top_k=top_k)

    llm_provider = OllamaLLMProvider()
    generator = Generator(llm_provider=llm_provider)

    return RAGPipeline(retriever=retriever, generator=generator)


async def get_metadata_db() -> MetadataDB:
    """メタデータDB の DI"""
    db = MetadataDB()
    await db.init()
    return db


@router.post("/", dependencies=[Depends(require_api_key)])
@limiter.limit(CHAT_RATE_LIMIT)
async def chat(
    request: Request,
    body: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    metadata_db: MetadataDB = Depends(get_metadata_db),
):
    """
    チャットエンドポイント（通常応答 or ストリーミング）
    system_prompt はコレクションに紐づいてサーバー側で管理。
    """
    _validate_collection_name(body.collection_name)

    # コレクションに紐づくシステムプロンプトをDBから取得
    system_prompt = await metadata_db.get_system_prompt(body.collection_name)

    if body.stream:
        # ストリーミング応答
        async def event_generator():
            async for event in pipeline.stream_query(
                body.question, body.collection_name, system_prompt or None
            ):
                if event["type"] == "chunk":
                    yield f"data: {json.dumps({'type': 'chunk', 'content': event['content']})}\n\n"
                elif event["type"] == "complete":
                    complete_data = event["content"]
                    source_files = list(
                        set(
                            c["source_file"]
                            for c in complete_data["source_chunks"]
                        )
                    )
                    yield f"data: {json.dumps({'type': 'complete', 'answer': complete_data['answer'], 'source_files': source_files})}\n\n"

        return StreamingResponse(
            event_generator(), media_type="text/event-stream"
        )
    else:
        # 通常応答
        response = await pipeline.query(body.question, body.collection_name, system_prompt or None)
        source_files = list(
            set(chunk.source_file for chunk in response.source_chunks)
        )
        return ChatResponse(answer=response.answer, source_files=source_files)
