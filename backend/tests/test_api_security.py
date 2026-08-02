"""APIの認証・CORS・レート制限・アップロード制限の回帰テスト。

ルート層のテストが無く、セキュリティ設定の壊れに気づけないため追加した。
app.main を import する前に環境変数を設定する必要がある（モジュール読み込み時に評価されるため）。
"""

import os

os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("CORS_ORIGINS", "https://example.test")
os.environ.setdefault("RATE_LIMIT_CHAT", "3/minute")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.api.routes import chat as chat_route  # noqa: E402
from app.core.rag.pipeline import RAGResponse  # noqa: E402

AUTH = {"X-API-Key": "test-key"}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class StubPipeline:
    """Ollama / Chroma を呼ばずにチャット経路だけ通すためのスタブ"""

    async def query(self, question, collection_name, system_prompt=None):
        return RAGResponse(answer="stub", source_chunks=[])


@pytest.fixture
def stub_pipeline():
    app.dependency_overrides[chat_route.get_rag_pipeline] = lambda: StubPipeline()
    yield
    app.dependency_overrides.clear()


def test_health_is_public(client):
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/collections/"),
        ("get", "/documents/default"),
    ],
)
def test_read_endpoints_require_api_key(client, method, path):
    assert getattr(client, method)(path).status_code == 403
    assert getattr(client, method)(path, headers=AUTH).status_code == 200


def test_chat_requires_api_key(client):
    r = client.post("/chat/", json={"question": "hi", "collection_name": "default"})
    assert r.status_code == 403


@pytest.mark.parametrize("collection", ["all", "shukatsu_private"])
def test_chat_rejects_collections_outside_allowlist(client, collection):
    r = client.post(
        "/chat/",
        json={"question": "hi", "collection_name": collection},
        headers=AUTH,
    )
    assert r.status_code == 403


def test_chat_rate_limit(client, stub_pipeline):
    codes = [
        client.post(
            "/chat/",
            json={"question": "hi", "collection_name": "default"},
            headers=AUTH,
        ).status_code
        for _ in range(5)
    ]
    assert codes[0] == 200
    assert 429 in codes


def test_cors_preflight_allows_configured_origin(client):
    r = client.options(
        "/chat/",
        headers={
            "Origin": "https://example.test",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-api-key",
        },
    )
    assert r.headers["access-control-allow-origin"] == "https://example.test"


def test_cors_preflight_rejects_unknown_origin(client):
    r = client.options(
        "/chat/",
        headers={
            "Origin": "https://evil.test",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in r.headers


def test_cors_does_not_allow_arbitrary_methods(client):
    r = client.options(
        "/chat/",
        headers={
            "Origin": "https://example.test",
            "Access-Control-Request-Method": "PATCH",
        },
    )
    assert "PATCH" not in r.headers.get("access-control-allow-methods", "")


def test_upload_rejects_oversized_file(client):
    r = client.post(
        "/documents/upload",
        files={"file": ("big.txt", b"x" * (11 * 1024 * 1024), "text/plain")},
        data={"collection_name": "default"},
        headers=AUTH,
    )
    assert r.status_code == 413


def test_upload_rejects_unsupported_extension(client):
    r = client.post(
        "/documents/upload",
        files={"file": ("bad.exe", b"x", "application/octet-stream")},
        data={"collection_name": "default"},
        headers=AUTH,
    )
    assert r.status_code == 400
