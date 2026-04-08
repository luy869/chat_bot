# RAG Platform

セルフホスト型の Retrieval-Augmented Generation プラットフォーム。

Markdown / PDF / テキストを投入し、ローカルLLM（Ollama）またはクラウドAPI（Gemini）で質問応答できる。
ポートフォリオサイトに埋め込むチャットBotとして実運用中。

**LangChain / LlamaIndex は一切使わず、各SDKを直接呼ぶ自前実装。**

---

## 技術スタック

| レイヤー | 技術 |
|---------|------|
| バックエンド | Python 3.12 + FastAPI |
| パッケージ管理 | uv |
| ベクトルDB | ChromaDB（コサイン距離） |
| LLM（Primary） | Ollama（qwen3.5:9b） |
| LLM（Fallback） | Gemini API（gemini-2.0-flash） |
| Embedding | Ollama（nomic-embed-text デフォルト / bge-m3 推奨） |
| メタデータDB | SQLite（aiosqlite） |
| フロントエンド | React + TypeScript + Vite + Tailwind CSS |
| コンテナ | Docker Compose |
| デプロイ | 自宅サーバー + Cloudflare Tunnel |

---

## アーキテクチャ

```
Frontend (React + Tailwind)
          |
      API (FastAPI)
      |- /chat              質問応答（SSEストリーミング対応）
      |- /documents         ドキュメント管理（APIキー認証）
      +- /collections       コレクション・システムプロンプト管理
          |
  Core (RAGパイプライン)
  |- Ingestion             チャンキング (Markdown/PDF/Text)
  |- Providers             LLM・Embedding抽象化 (ABC -> Ollama/Gemini)
  |- VectorStore           ChromaDB ラッパー
  +- RAG                   Retriever -> Generator
          |
   Backends
   |- ChromaDB             ベクトル検索（組み込みモード）
   |- SQLite               メタデータ・システムプロンプト管理
   |- Ollama               ローカルLLM + Embedding
   +- Gemini API           クラウドLLM（フォールバック）
```

---

## 機能

- **複数形式のドキュメント投入** — Markdown（見出しベースチャンキング + heading_path）、PDF（ページ単位）、プレーンテキスト（段落単位）
- **ベクトル検索** — ChromaDB でコサイン距離検索（Embedding モデルは環境変数で切り替え可）
- **SSEストリーミング** — トークン単位のリアルタイム応答
- **LLMプロバイダ差し替え** — ABCパターンで Ollama / Gemini を透過的に切り替え
- **コレクション別システムプロンプト** — サーバーサイドで管理し、クライアントからの改ざんを防止
- **API認証** — `X-API-Key` ヘッダーで書き込みAPIを保護
- **マルチGPU対応** — 環境変数で GPU 割り当て・並列数を制御

---

## クイックスタート

### 前提条件

- Docker & Docker Compose
- NVIDIA GPU + nvidia-container-toolkit（Ollama用）

### セットアップ

```bash
git clone https://github.com/luy869/rag-platform.git
cd rag-platform

# 環境変数を設定
cp .env.example .env  # 必要に応じて編集

# 起動（初回はモデルのダウンロードに時間がかかる）
docker compose up --build
```

起動後:
- **Backend**: http://localhost:4000
- **Frontend**: http://localhost:3000
- **Ollama**: http://localhost:11434

### 動作確認

```bash
# ヘルスチェック
curl http://localhost:4000/health

# ドキュメント投入
curl -X POST http://localhost:4000/documents/upload \
  -H "X-API-Key: your_api_key" \
  -F "file=@example.md" \
  -F "collection_name=default"

# 質問（非ストリーミング）
curl -X POST http://localhost:4000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"question": "RAGとは何ですか？", "collection_name": "default"}'

# 質問（ストリーミング）
curl -N -X POST http://localhost:4000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"question": "RAGとは何ですか？", "collection_name": "default", "stream": true}'
```

---

## ポートフォリオBot

`portfolio-bot/` にプロフィールデータ投入スクリプトがある。

```bash
# プロフィールデータを投入
cd portfolio-bot
python ingest.py

# システムプロンプトを設定
curl -X PUT http://localhost:4000/collections/portfolio/system-prompt \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"system_prompt": "'"$(cat system_prompt.md)"'"}'
```

---

## API リファレンス

### POST /chat/

質問応答。`stream: true` で SSE ストリーミング。

```
Request:  {"question": "...", "collection_name": "default", "stream": false}
Response: {"answer": "...", "source_files": ["example.md"]}
```

ストリーミング時は `text/event-stream` で以下のイベントを返す:
```
data: {"type": "chunk", "content": "こん"}

data: {"type": "complete", "answer": "...", "source_files": [...]}
```

### POST /documents/upload (認証必須)

ドキュメント投入。`multipart/form-data` で送信。

```
Form: file=@example.md, collection_name=default
Response: {"status": "success", "document_id": "uuid-xxx", "chunk_count": 5}
```

### DELETE /documents/{document_id} (認証必須)

ドキュメント削除。ChromaDB とメタデータ DB の両方からクリーンアップ。

### GET /collections/

コレクション一覧取得。

### POST /collections/{name} (認証必須)

コレクション作成。

### DELETE /collections/{name} (認証必須)

コレクション削除。

### PUT /collections/{name}/system-prompt (認証必須)

コレクション別システムプロンプト設定。

### GET /health

ヘルスチェック。

---

## ディレクトリ構成

```
rag-platform/
├── README.md
├── CLAUDE.md                 プロジェクト指針
├── PLAN.md                   開発計画書
├── DEVELOPMENT_LOG.md        開発ログ（面接用）
├── RAG_TEXTBOOK.md           技術解説教科書
├── docker-compose.yaml
│
├── backend/
│   ├── dockerfile
│   ├── pyproject.toml        uv 依存管理
│   ├── uv.lock
│   ├── app/
│   │   ├── main.py           FastAPI エントリポイント
│   │   ├── config.py         設定管理
│   │   ├── api/
│   │   │   ├── auth.py       API キー認証
│   │   │   └── routes/
│   │   │       ├── chat.py          チャット（SSE対応）
│   │   │       ├── documents.py     ドキュメント管理
│   │   │       └── collections.py   コレクション管理
│   │   ├── core/
│   │   │   ├── ingestion/
│   │   │   │   ├── markdown.py      見出しベースチャンキング
│   │   │   │   ├── pdf.py           ページ単位チャンキング
│   │   │   │   └── text.py          段落単位チャンキング
│   │   │   ├── providers/
│   │   │   │   ├── base.py          ABC（LLMProvider / EmbeddingProvider）
│   │   │   │   ├── ollama.py        Ollama 実装
│   │   │   │   └── gemini.py        Gemini 実装
│   │   │   ├── vectorstore/
│   │   │   │   └── chroma.py        ChromaDB ラッパー
│   │   │   └── rag/
│   │   │       ├── pipeline.py      RAG パイプライン
│   │   │       ├── retriever.py     ベクトル検索
│   │   │       └── generator.py     LLM 回答生成
│   │   ├── db/
│   │   │   └── metadata.py          SQLite メタデータ管理
│   │   └── models/
│   │       └── chunk.py             Chunk データクラス
│   └── tests/                       ユニットテスト
│
├── frontend/                  React UI
├── portfolio-bot/             プロフィール投入スクリプト
│   ├── profile.md             プロフィールデータ
│   ├── system_prompt.md       ポートフォリオBot用プロンプト
│   ├── ingest.py              投入スクリプト
│   └── ingest_default.py      デフォルトコレクション用
│
└── docker-compose.yaml
```

---

## 環境変数

`.env` を作成して設定:

```env
# Ollama
OLLAMA_HOST=http://ollama:11434
OLLAMA_EMBED_MODEL=bge-m3          # デフォルト: nomic-embed-text（日本語精度を上げる場合は bge-m3 を推奨）
                                   # bge-m3 を使う場合は事前に: ollama pull bge-m3

# GPU 設定
GPU_DEVICES=0                      # 使用する GPU（マルチ GPU: 0,1）
OLLAMA_PARALLEL=1                  # Ollama 同時リクエスト数

# API 認証
API_KEY=your_secret_key            # 未設定で認証スキップ（開発用）

# CORS
CORS_ORIGINS=http://localhost:3000 # カンマ区切りで複数指定可

# Gemini（オプション）
GEMINI_API_KEY=your_api_key
```

---

## 開発

### バックエンドのテスト

```bash
cd backend
uv run pytest -v
```

### パッケージ追加

```bash
cd backend
uv add <package>
```

### Docker の注意事項

- `.venv` は anonymous volume で除外する（ホストマウントとの競合を防ぐ）
- `pyproject.toml` の `requires-python`、`.python-version`、Dockerfile の Python バージョンを必ず統一する

---

## 設計上の判断

| 判断 | 理由 |
|------|------|
| 自前実装 > LangChain | ブラックボックス排除、面接で全部説明可能、依存最小化 |
| FastAPI > Flask | ネイティブ非同期、SSE、Pydantic 統合 |
| ChromaDB > Qdrant | 組み込みモード、追加サービス不要 |
| SSE > WebSocket | 一方向通信で十分、実装がシンプル |
| 見出しベース > 固定長チャンキング | 文書構造を活用、heading_path でコンテキスト保持 |
| bge-m3 > nomic-embed-text | 日本語セマンティック検索の精度（`OLLAMA_EMBED_MODEL` で切り替え） |
| SQLite > PostgreSQL | ゼロ設定、メタデータ規模に十分 |
| chat() > generate() | メッセージロール分離でプロンプトインジェクション対策 |

---

## ライセンス

MIT
