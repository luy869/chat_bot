import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from app.api.routes import chat, documents, collections
from app.core.rate_limit import limiter

app = FastAPI(title="RAG Platform", version="0.1.0")

# レート制限（アプリ層）。Cloudflare 側の制御が実装されるまでの最終防衛ライン
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS設定
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
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