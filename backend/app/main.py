from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import chat, documents, collections

app = FastAPI(title="RAG Platform", version="0.1.0")

# CORS設定（luy869.net + localhost のみ）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://luy869.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーター登録
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(collections.router)


@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok"}