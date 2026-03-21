from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from app.core.rag.pipeline import RAGPipeline, RAGResponse
from app.core.rag.retriever import Retriever
from app.core.rag.generator import Generator

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """チャットリクエスト"""

    question: str
    collection_name: str = "default"
    stream: bool = False
    system_prompt: str | None = None


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


@router.post("/")
async def chat(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """
    チャットエンドポイント（通常応答 or ストリーミング）

    Args:
        request: ChatRequest（質問、コレクション、ストリーミングフラグ）

    Returns:
        ストリーミングの場合：Server-Sent Events (SSE) ストリーム
        通常応答の場合：JSON レスポンス
    """
    if request.stream:
        # ストリーミング応答
        async def event_generator():
            async for event in pipeline.stream_query(
                request.question, request.collection_name, request.system_prompt
            ):
                if event["type"] == "chunk":
                    # テキストの断片をSSEで送信
                    yield f"data: {json.dumps({'type': 'chunk', 'content': event['content']})}\n\n"
                elif event["type"] == "complete":
                    # ストリーム完了時に参照チャンク情報を送信
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
        # 通常応答（全体を待ってから返す）
        response = await pipeline.query(request.question, request.collection_name, request.system_prompt)
        source_files = list(
            set(chunk.source_file for chunk in response.source_chunks)
        )
        return ChatResponse(answer=response.answer, source_files=source_files)
