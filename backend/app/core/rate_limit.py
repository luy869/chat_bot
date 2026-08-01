import os
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def get_client_ip(request: Request) -> str:
    """クライアントの実IPを取得する。

    本番はホスト nginx / Cloudflare Tunnel の後ろで動くため、
    request.client.host は素通しのプロキシのIPになりがちで
    IPベースのレート制限が実質意味を失う。
    Cloudflare が付与する CF-Connecting-IP（エッジで上書きされるためクライアントは偽装不可）を
    優先し、無ければ X-Forwarded-For の先頭、それも無ければ通常の remote address を使う。
    """
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip

    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    return get_remote_address(request)


# レート制限値は環境変数で調整可能にしておく（デフォルトは PLAN.md の想定値）
CHAT_RATE_LIMIT = os.getenv("RATE_LIMIT_CHAT", "5/minute")
UPLOAD_RATE_LIMIT = os.getenv("RATE_LIMIT_UPLOAD", "10/minute")

limiter = Limiter(key_func=get_client_ip)
