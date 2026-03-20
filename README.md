# RAG Platform

セルフホスト型のRetrieval-Augmented Generationプラットフォーム。

ドキュメント（Markdown、PDF、テキスト）を投入し、ローカルLLM（Ollama）またはクラウドAPI（Gemini）で質問応答できます。

---

## 機能

- ✅ **複数形式のドキュメント投入** — Markdown（見出し構造を活用）、PDF（ページ単位）、プレーンテキスト
- ✅ **自動チャンキング** — ドキュメント形式に応じた最適な分割
- ✅ **ベクトル検索** — ChromaDB で関連チャンク検索
- ✅ **RAG生成** — 検索結果をコンテキストに含めてLLMで回答生成
- ✅ **ストリーミング応答** — SSE で生成中の回答をリアルタイム表示
- ✅ **LLM差し替え可能** — Ollama（ローカル）と Gemini API（クラウド）を透過的に切り替え

---

## アーキテクチャ

```
Frontend (React + Tailwind)
          ↓
      API (FastAPI)
      ├─ /chat              質問応答
      ├─ /documents         ドキュメント管理
      └─ /collections       コレクション管理
          ↓
  Core (RAGパイプライン)
  ├─ Ingestion             チャンキング (Markdown/PDF/Text)
  ├─ Providers            LLM・埋め込み抽象化 (Ollama/Gemini)
  ├─ VectorStore          ChromaDB ラッパー
  └─ RAG                  Retriever → Generator
          ↓
   Backends
   ├─ ChromaDB            ベクトル検索
   ├─ SQLite              メタデータ管理
   ├─ Ollama              ローカルLLM（デフォルト）
   └─ Gemini API          クラウドLLM（フォールバック）
```

---

## クイックスタート

### 前提

- Docker & Docker Compose
- Python 3.12（ローカルテスト用）
- Node.js（フロントエンド開発用）

### セットアップ

```bash
# リポジトリクローン
git clone https://github.com/luy869/rag-platform.git
cd rag-platform

# Docker Compose で起動
docker compose up --build
```

起動後：
- **Backend**: http://localhost:4000
- **Frontend**: http://localhost:3000
- **Ollama**: http://localhost:11434
- **ChromaDB**: http://localhost:8000

### 動作確認

```bash
# ドキュメント投入
curl -X POST http://localhost:4000/documents/upload \
  -F "file=@example.md" \
  -F "collection_name=default"

# 質問
curl -X POST http://localhost:4000/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is RAG?",
    "collection_name": "default",
    "stream": false
  }'
```

---

## 開発

### バックエンドのテスト

```bash
cd backend
uv run pytest -v
```

全31テストが成功します。

### パッケージ追加

```bash
cd backend
uv add <package>
```

---

## 技術スタック

| レイヤー | 技術 |
|---------|------|
| Backend | Python 3.12 + FastAPI + aiosqlite |
| Vector DB | ChromaDB |
| LLM (Primary) | Ollama（gemma3:12b等） |
| LLM (Fallback) | Gemini API |
| Embedding | Ollama (nomic-embed-text) |
| Frontend | React + TypeScript + Tailwind CSS |
| Container | Docker Compose |
| Metadata | SQLite |

### フレームワーク非依存

LangChain や LlamaIndex を使わず、各プロバイダの公式SDKを直接使用しています。

**メリット**：
- 面接で「RAGパイプラインの中身を全部理解している」と説明可能
- 依存最小化
- デバッグ容易

---

## API リファレンス

### POST /documents/upload

ドキュメント投入

```json
{
  "file": "example.md",
  "collection_name": "default"
}

Response:
{
  "status": "success",
  "document_id": "uuid-xxx",
  "chunk_count": 5
}
```

### POST /chat/

質問応答

```json
{
  "question": "What is RAG?",
  "collection_name": "default",
  "stream": false
}

Response:
{
  "answer": "RAG is a technique...",
  "source_files": ["example.md"]
}
```

ストリーミング (stream=true) では Server-Sent Events (SSE) で応答を段階的に配信します。

### GET /collections/

コレクション一覧

```json
[
  {
    "name": "default",
    "document_count": 2,
    "created_at": "2026-03-21T00:00:00"
  }
]
```

---

## ディレクトリ構成

```
rag-platform/
├── README.md                 このファイル
├── CLAUDE.md                プロジェクト指針
├── PLAN.md                  詳細な開発計画書
│
├── backend/
│   ├── pyproject.toml
│   ├── dockerfile
│   ├── app/
│   │   ├── main.py          FastAPI アプリケーション
│   │   ├── config.py        設定
│   │   ├── api/
│   │   │   └── routes/      APIエンドポイント
│   │   ├── core/
│   │   │   ├── ingestion/   チャンキング
│   │   │   ├── providers/   LLM・Embedding抽象化
│   │   │   ├── vectorstore/ ChromaDB ラッパー
│   │   │   └── rag/         RAGパイプライン
│   │   ├── db/              メタデータ管理
│   │   └── models/          データクラス
│   └── tests/               ユニットテスト (31個)
│
├── frontend/
│   └── index.html           React UI
│
└── docker-compose.yaml      マルチコンテナ構成
```

---

## 環境変数

`.env` を作成して設定：

```env
# Ollama
OLLAMA_HOST=http://ollama:11434
OLLAMA_MODEL=gemma3:12b

# Gemini (オプション、Ollama が利用不可の場合のフォールバック)
GEMINI_API_KEY=your_api_key_here

# ChromaDB
CHROMA_DB_PATH=./chroma_data

# FastAPI (カンマ区切り、デフォルト: http://localhost:3000)
CORS_ORIGINS=http://localhost:3000,https://example.com
```

---

## 拡張ポイント

**ロードマップ**（優先度順）

1. **スコアフィルタリング** — 類似度が低いチャンクを除外
2. **プロンプトエンジニアリング** — Few-shot examples、言語・トーン指定
3. **ハイブリッド検索** — ベクトル検索 + BM25 キーワード検索
4. **チャンク再ランキング** — LLM で関連度を再評価
5. **質問の言い換え** — 複数表現で検索
6. **エージェント化** — 複雑な質問に複数検索→統合で対応

---

## トラブルシューティング

### Ollama が接続できない

```bash
# Ollama が起動しているか確認
curl http://localhost:11434/api/models

# 起動していなければ
ollama serve
```

### ChromaDB がスペースを消費

```bash
# チャンクデータを削除
rm -rf ./chroma_data
```

### メモリ不足

Ollama のメモリを制限：

```yaml
# docker-compose.yaml
ollama:
  environment:
    - OLLAMA_NUM_GPU=1
    - OLLAMA_MAX_LOADED_MODELS=1
```

---

## ライセンス

MIT
