from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from app.core.rag.pipeline import RAGPipeline, RAGResponse
from app.core.rag.retriever import Retriever
from app.core.rag.generator import Generator
from app.db.metadata import MetadataDB

router = APIRouter(prefix="/chat", tags=["chat"])


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
    retriever = Retriever(vectorstore=vectorstore)

    llm_provider = OllamaLLMProvider()
    generator = Generator(llm_provider=llm_provider)

    return RAGPipeline(retriever=retriever, generator=generator)


async def get_metadata_db() -> MetadataDB:
    """メタデータDB の DI"""
    db = MetadataDB()
    await db.init()
    return db


@router.post("/")
async def chat(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    metadata_db: MetadataDB = Depends(get_metadata_db),
):
    """
    チャットエンドポイント（通常応答 or ストリーミング）
    system_prompt はコレクションに紐づいてサーバー側で管理。
    """
    # コレクションに紐づくシステムプロンプトをDBから取得
    system_prompt = await metadata_db.get_system_prompt(request.collection_name)

    if request.stream:
        # ストリーミング応答
        async def event_generator():
            async for event in pipeline.stream_query(
                request.question, request.collection_name, system_prompt or None
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
        response = await pipeline.query(request.question, request.collection_name, system_prompt or None)
        source_files = list(
            set(chunk.source_file for chunk in response.source_chunks)
        )
        return ChatResponse(answer=response.answer, source_files=source_files)
