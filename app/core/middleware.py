import re
from urllib.parse import urlparse

from core.config import HOST_CHECK_ENABLED
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


def is_ip_address(host: str) -> bool:
    return re.match(r"^\d+\.\d+\.\d+\.\d+$", host) is not None


class HostCheckMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if not HOST_CHECK_ENABLED:
            return await call_next(request)

        host = request.url.hostname
        if host and is_ip_address(host):
            return JSONResponse({"error": "Access denied"}, status_code=403)
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """セキュリティヘッダーを追加するミドルウェア"""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        # XSS対策
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        # Content Security Policy (インラインスクリプトを許可しつつ外部スクリプトを制限)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "frame-ancestors 'none'"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """
    CSRF保護ミドルウェア
    - フォーム送信（Content-Type: application/x-www-form-urlencoded）の場合はRefererチェック
    - AJAX（Content-Type: application/json または X-Requested-With ヘッダー）は許可
    """

    SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}

    async def dispatch(self, request, call_next):
        if request.method in self.SAFE_METHODS:
            return await call_next(request)

        # AJAXリクエスト（X-Requested-Withヘッダー付き）は許可
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return await call_next(request)

        # Content-Type が application/json の場合は許可（ブラウザのフォームからは送信不可）
        content_type = request.headers.get("Content-Type", "")
        if "application/json" in content_type:
            return await call_next(request)

        # フォーム送信の場合はReferer/Originをチェック
        origin = request.headers.get("Origin")
        referer = request.headers.get("Referer")
        host = request.headers.get("Host")

        # OriginまたはRefererが存在し、同一オリジンかチェック
        if origin:
            # Originヘッダーからホスト部分を抽出
            origin_host = urlparse(origin).netloc
            if origin_host != host:
                return JSONResponse({"error": "CSRF validation failed"}, status_code=403)
        elif referer:
            referer_host = urlparse(referer).netloc
            if referer_host != host:
                return JSONResponse({"error": "CSRF validation failed"}, status_code=403)
        else:
            # Origin も Referer もない場合は拒否
            return JSONResponse({"error": "CSRF validation failed: missing origin"}, status_code=403)

        return await call_next(request)
