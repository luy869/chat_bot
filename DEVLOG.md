# rag-platform 開発ログ

意思決定・設計判断・トラブルシューティングを時系列で記録する。

---

## 2026-07-31 — DEVLOG 運用開始

これ以前の開発（設計判断・トラブルシューティング）は記録されていない。
過去分を後から書き起こすと推測が混ざるため、**ここから先の分のみ記録する**。

以降、以下を書く:

- 技術選定と、**採用しなかった選択肢とその理由**
  （埋め込みモデル、ベクトルDB、チャンク戦略など、RAGは選択肢が多く比較の記録が効く）
- 詰まった問題の「症状 → 原因 → 修正」
- 検索精度を変えた施策と、その前後の差

※ 面接で問われるのは「なぜその設計にしたか」なので、**捨てた案を残すことに価値がある**。

---

## 2026-08-01 — 就活用プライベートコレクション（`shukatsu_private`）の追加

`/home/luy869/ES/`（就活ワークスペース、読み取り専用）の過去ES・自己分析・
面接振り返りを検索できるプライベートコレクションを追加した
（詳細: `job-hunting-rag/README.md`）。

### コレクション分離の方法
- 公開Botは `luy_web/widget/src/ChatWidget.tsx` の `COLLECTION_NAME = "default"`
  にハードコードされており、コレクション選択UIを持たない。新設した
  `shukatsu_private` はこの経路には現れない。
- ただし調べた結果、`GET /collections/`・`POST /chat/` はどちらも認証なしで、
  `chat` の `collection_name` はクライアント指定・許可リスト検証なしという
  既存の設計上の隙間を発見した（`shukatsu_private` 固有の問題ではなく元からある）。
  今回は「公開ウィジェットの挙動を変更しない」というスコープのため
  `collections.py`/`chat.py` は変更していない。本番サーバーに同期する場合は
  先に認証・許可リストを入れる必要がある、と判断した。

### 埋め込みモデル選定
- 埋め込みは `bge-m3`（1024次元）を明示指定。既存 `default` コレクションは
  `nomic-embed-text`（768次元）のまま変更していない。
- **採用しなかった選択肢**: サーバーの環境変数 `OLLAMA_EMBED_MODEL` を
  `bge-m3` に変更する方法。DI (`documents.py`/`collections.py`) が
  この環境変数を都度読むため、変更すると同一プロセスの全コレクションの
  埋め込みが切り替わり、`default`（768次元）への問い合わせが次元不一致で
  壊れる。そのため HTTP API を経由せず、`app.core` のクラスを直接呼ぶ
  スクリプト（`job-hunting-rag/ingest.py`）にして、埋め込みモデルを
  呼び出し側で明示指定する方式にした。

### チャンク戦略と精度調整
- 既存の `MarkdownChunker`（見出しベース、`heading_path`保持）をそのまま
  利用。ES回答は見出しが設問文そのもの（例:「## 学生時代に最も打ち込んだこと」）
  になっているファイルが多く、相性が良かった。
- **問題**: `episodes.md` の「### 使える質問タイプ」チェックリスト
  （設問名をそのまま列挙）が、対応する設問クエリで埋め込み類似度が
  不当に高くなり、概要/課題/行動/結果といった本文チャンクより上位に
  来てしまう問題を検索テストで確認。
  **修正**: 投入前にこのセクションだけを除去するプリプロセスを追加。
  修正後、実際のES回答や本文チャンクが上位に来るようになった。
- **問題**: 面接レポート（74件）の文字起こしは `SPEAKER_00/01/02` という
  ASRの自動診断ラベルだが、同一ラベル内に別人の発言が混在するケースを
  実際に確認した（`DeNA 面接_report.md`: 面接官の発言と本人の自己紹介が
  同一タグに連結）。機械的に安全な話者分離ができないため、「無理に切り
  分けない」方針で文字起こし本文は74件全て投入対象から除外し、
  AIが生成したスコア・良かった点・改善点・総合コメントのみを投入した。
- 距離指標はコサイン距離（`ChromaVectorStore`が`hnsw:space: cosine`を
  明示指定済み、コードで確認）。過去の「L2距離のまま閾値設定→全件
  フィルタアウト」事故は再現していない。

### 検証結果（4問）
- 「学生時代に力を入れたことを400字で」「あなたの強みは」→ 的中
  （ES回答・準備シートのQ&Aがそのまま1位）
- 「困難を乗り越えた経験」→ 概ね的中（ノイズが4-5位に軽度混入）
- 「なぜその技術を選んだのか」→ 半分的中。「選んだ」という動詞が
  企業選定理由（志望動機）とも一致するため、上位が志望動機系に偏る。
  クエリに「技術選定理由」を補うと目的のチャンクの順位が上がることを確認。
  将来の課題としてクエリ拡張/リランキングを検討する余地がある。

### 環境構築の副産物
- 開発機のDockerブリッジ（`172.17.0.1:11434`）からホストのOllamaに到達
  できず（タイムアウト）、ホストのOllamaを直接使う構成にした。
- `backend/chroma_data/`・`backend/metadata.db` が過去の `docker compose up`
  実行時にroot所有になっており、ホストユーザーから書き込めなかった。
  使い捨てのalpineコンテナで `chown` して解決。

---

## 2026-08-01 — セキュリティレビューと修正（`shukatsu_private` 追加を受けての棚卸し）

### 症状
`shukatsu_private` 追加時に見つけていた「`GET /collections/`・`POST /chat/` が無認証」
「`chat` の `collection_name` に許可リストがない」という隙間を放置したままにできない
状態になった（プライベートコレクションが同一プロセス上に存在する以上、この隙間は
「情報漏洩の実害」に直結する）ため、あらためて全体をレビューして修正した。

### 原因
1. `POST /chat/`・`GET /collections/`・`GET /documents/{collection_name}` に
   認証依存 (`require_api_key`) が付いていなかった。Issue 28（旧
   `DEVELOPMENT_LOG.md`）で認証を入れたのは「変更系」エンドポイントのみで、
   参照系は対象外だった。
2. `chat` の `collection_name` はクライアントが自由な文字列を送れ、
   `"all"` を渡すと `ChromaVectorStore._search_all_collections` が
   **存在する全コレクションを横断検索**する実装だった
   （`backend/app/core/vectorstore/chroma.py`）。許可リストが一切なかった。
3. もっと根が深い問題として、`luy_web/worker/src/index.js` は
   「Origin ヘッダーが `https://luy869.net` と一致するかどうか」だけで
   リクエストを許可し、通れば **すべての `/api/*` パスに対して**
   `X-API-Key` を自動付与していた。Origin ヘッダーはブラウザ以外の
   クライアント（curl等）からは自由に偽装できるため
   （`curl -H "Origin: https://luy869.net" https://luy869.net/api/collections/` が
   200 を返すことを実測で確認済み）、この設計だと「バックエンドの
   `require_api_key` を素通しできる＝誰でも管理系エンドポイント
   （`documents/upload`・`collections` の作成/削除・system-prompt変更等）を
   叩ける」状態だった。つまり `require_api_key` を付けるだけでは
   Worker 経由の攻撃は防げず、**API キーの有無は「正規の呼び出し元」を
   意味しない**ことが判明した。

### 修正
1. `POST /chat/`・`GET /collections/`・`GET /documents/{collection_name}` に
   `Depends(require_api_key)` を追加（`backend/app/api/routes/*.py`）。
2. `chat.py` に `collection_name` の許可リストを追加
   （env `PUBLIC_CHAT_COLLECTIONS`, デフォルト `"default"` のみ）。
   **API キーの有無に関わらず無条件で適用**する設計にした。理由は上記3の
   とおり、Worker がキーを一律自動付与するため「キーがあるから許可」という
   分岐は攻撃者にも同じキーが渡ってしまい意味をなさないため。
   `"all"` は常に拒否される。
3. `luy_web/worker/src/index.js`（別リポジトリ）に転送パスのホワイトリストを
   追加。`POST /api/chat/` と `GET /api/health` 以外は Worker の時点で 403 を
   返し、`documents`/`collections` 系はそもそも Worker から到達できないようにした。
   これが実質的な一番効いている修正だと考えている。
   ローカルコミットのみ（本番未反映）。
4. `slowapi` でアプリ層のレート制限を追加（`backend/app/core/rate_limit.py`）。
   `/chat/` は `5/minute`、`/documents/upload` は `10/minute`（env で変更可）。
   Cloudflare Tunnel の背後で動くため `request.client.host` は素通しの
   プロキシIPになり得る。`CF-Connecting-IP`（Cloudflareがエッジで上書きする
   ため偽装不可）→ `X-Forwarded-For` → 通常の remote address の順で
   実クライアントIPを解決するキー関数にした。
5. `documents.py` のアップロードでファイルサイズ上限を追加
   （デフォルト10MB、env `MAX_UPLOAD_SIZE_BYTES` で変更可）。
   `await file.read()` で無制限に読み込むと巨大ファイルでメモリを
   食い潰すDoSになるため、1MBずつ読みながら上限超過で即座に打ち切る方式にした。
   拡張子チェックも読み込み前に移動し、無駄なI/O・コレクション作成をしない。
6. `.gitignore`（ルート/`backend/`）に `.env` を追加（元々漏れていた）。

### 採用しなかった選択肢
- **チャットエンドポイントに「有効なAPIキーなら `all` も許可」という分岐**:
  上記の理由（Workerが誰にでも同じキーを渡してしまう）で見送った。
  本当に横断検索したい場合は `job-hunting-rag/search_test.py` のように
  HTTP を経由せず `app.core` を直接呼ぶ方式にすべきと判断。
- **Cloudflare Access の導入**: プライベートコレクションを完全に別経路
  （別ポート/別Tunnel/Cloudflare Access）に分離する案は効果が大きい一方
  設計変更の影響範囲が大きいため、今回は実装せず設計案として提示に留めた
  （詳細は報告参照）。
- **依存ライブラリの一括アップデート**（`pypdf`/`starlette`/`chromadb` に
  既知脆弱性あり、`pip-audit` で確認）: 特に `pypdf` はメジャーバージョンが
  複数上がっており、`fastapi[standard]` 経由の `starlette` も含めて
  互換性の検証なしに上げるとテストなしで壊すリスクが高いため見送り、
  一覧化して報告のみに留めた。

### 検証方法
ローカルで `API_KEY` 未設定/設定の両方で `uvicorn` を起動し、
公開Botと同じリクエスト（`collection_name: "default"`）が
認証・許可リスト・レート制限の各レイヤーを通過して実際の
RAGパイプライン（Ollama呼び出し）まで到達することを確認した
（この環境にOllamaモデルが無いため最終的に500になるが、それは
今回の変更と無関係なインフラ起因と切り分け済み）。
あわせて `collection_name: "all"` / `"shukatsu_private"` が
キーの有無に関わらず403になること、無認証の `/collections/`・
`/documents/{name}` が403になること、`/chat/` への6回目以降の
リクエストが429になること、10MB超のアップロードが413になることを確認した。

---

## 2026-08-02 — 依存の脆弱性一掃、CORS の絞り込み、ルート層テストの追加

前日のセキュリティレビューで「未修正」として残した項目のうち、依存更新・CORS・
nginx・ウィジェットの Markdown レンダリングを処理した。

### 依存ライブラリ（50件 → 1件）

`pip-audit` を lock ファイル（`uv export`）に対して実行した。

- 最初は `uv run pip-audit` がシステム側の Python 環境（cloud-init、twisted 等の
  Ubuntu パッケージ）を監査していて数字が合わなかった。プロジェクトの Python は
  3.12 で、pip-audit を起動したインタプリタが 3.10 だったため、pip-audit が内部で
  作る一時 venv で `numpy==2.4.3` が解決できず失敗する問題もあった。
  `uv run --with pip-audit python -m pip_audit -r <export> --no-deps` で解消
- 直接依存で上げたのは `pypdf`（6.9.1 → 6.14.2、19件）のみ。残りは `uv lock --upgrade`
  による推移的依存の更新で解消した
- 残る1件は `chromadb` の PYSEC-2026-311（修正版なし）。これは Chroma の HTTP サーバー
  API（`/api/v2/.../collections` に `trust_remote_code` を送る経路）が対象で、
  本アプリは `chromadb.PersistentClient` によるプロセス内埋め込みで動いており
  当該エンドポイントを公開していないため、この構成では踏めない

### starlette 0.52 → 1.3.1（メジャー更新）

前日は「メジャーが上がるため見送り」としていたが、今回の一括更新に含まれた。
問題はテスト側にあった。**ルート層のテストが1件も無く**、この更新で認証・レート制限・
CORS が壊れても気づけない状態だった（既存31件はチャンカーとプロバイダのユニットテスト）。

`tests/test_api_security.py` を追加し、以下を回帰テスト化した。

- `/collections/`・`/documents/{name}` の API キー必須
- `/chat/` の API キー必須と、コレクション許可リスト（`all` / `shukatsu_private` は403）
- レート制限が発火すること（slowapi が starlette 1.x で動くかの確認を兼ねる）
- CORS プリフライト（許可オリジンは通り、未知オリジンと PATCH は通らない）
- アップロードの 10MB 超 → 413、拡張子違反 → 400

チャットのテストでは `get_rag_pipeline` を `dependency_overrides` で差し替えている。
実物を使うと Ollama と Chroma に到達してしまい、テストが環境依存になるため。

結果は 39 passed / 4 failed。失敗4件（markdown_chunker 2件・providers 2件）は
この変更の前から同じ内容で失敗しており、今回の更新による退行ではない。

### CORS

`allow_methods` / `allow_headers` がワイルドカードだったので、実際に使う
`GET/POST/PUT/DELETE/OPTIONS` と `Content-Type` / `X-API-Key` に絞った。
`allow_origins` は元から絞られていたので実害は無かったが、プリフライトの
応答で「何でも受け付ける」と宣言する必要はない。

### Ollama の keep_alive

`OLLAMA_KEEP_ALIVE` で設定できるようにしたが、**既定では送らない**（Ollama 既定の5分のまま）。
提案書には「24h にすれば初回応答10秒問題が解決する」とあったが、この GPU は他サービスと
共有しているため、モデルを常駐させると他が動かせなくなる。長く保持したい場合だけ
明示的に設定する。ホスト側の systemd（Ollama はコンテナではない）は sudo が必要で
触れないため、アプリから `keep_alive` を渡す経路のみを用意した。

### nginx.conf

`client_max_body_size 100M` をアプリ側の上限（10MB）に合わせた。
なお本番でこのファイルは**使われていない**。本番の `rag-platform` コンテナは
uvicorn が 0.0.0.0:8000 で直接待ち受けており、nginx コンテナは存在しない
（`docker ps` で確認。`nginx.conf` を参照するのは `docker-compose.yaml` の
ローカル開発用フロントエンド配信のみ）。

### ウィジェットの Markdown レンダリング（対応不要と判断）

`markdown-to-jsx` が LLM 出力をそのまま描画している点を実測で検証した。
9.8.2 + React の組み合わせで以下はすべて無害化される。

| 入力 | 描画結果 |
|---|---|
| `<script>...</script>` | `<span>&lt;script&gt;</span>`（テキストとしてエスケープ） |
| `<img src=x onerror=...>` | `<img src="x"/>`（イベントハンドラが落ちる） |
| `<svg onload=...>` / `<div onclick=...>` | 属性が落ちる |
| `[click](javascript:...)` / `JaVaScRiPt:` / `data:text/html,...` | `<a>click</a>`（href が付かない） |
| `[ok](https://example.com)` | 正常にリンクされる |

React が小文字のイベントハンドラ属性を拒否し、markdown-to-jsx 側のサニタイザが
`javascript:` / `data:` スキームを落とすため。追加のサニタイズは入れない。
