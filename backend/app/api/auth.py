import os
from fastapi import HTTPException, Header


async def require_api_key(x_api_key: str = Header(None)):
    """
    APIキー認証。変更系エンドポイント（アップロード・削除等）に適用。
    API_KEY 環境変数が未設定の場合は認証をスキップ（開発環境用）。
    """
    expected = os.getenv("API_KEY", "")
    if not expected:
        return  # 開発環境: API_KEY未設定なら認証スキップ
    if x_api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API key")
