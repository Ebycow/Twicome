from fastapi import FastAPI
from fastapi.responses import JSONResponse

from core.config import FAISS_API_URL, ROOT_PATH
from core.middleware import CSRFProtectionMiddleware, HostCheckMiddleware, SecurityHeadersMiddleware
from faiss_search import ping_faiss_api
from routers import ALL_ROUTERS

app = FastAPI(root_path=ROOT_PATH)
app.add_middleware(CSRFProtectionMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(HostCheckMiddleware)


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


@app.on_event("startup")
def check_faiss_api():
    """起動時に faiss-api への接続確認"""
    if FAISS_API_URL:
        try:
            ping_faiss_api()
            print(f"[faiss] faiss-api 接続確認完了: {FAISS_API_URL}")
        except Exception as e:
            print(f"[faiss] Warning: {e}")
            print("[faiss] 埋め込み検索機能は利用できません")
    else:
        print("[faiss] FAISS_API_URL 未設定 - 埋め込み検索機能は無効")


for router in ALL_ROUTERS:
    app.include_router(router)
