from fastapi import FastAPI

from core.config import ROOT_PATH
from core.middleware import CSRFProtectionMiddleware, HostCheckMiddleware, SecurityHeadersMiddleware
from faiss_search import get_model
from routers import ALL_ROUTERS

app = FastAPI(root_path=ROOT_PATH)
app.add_middleware(CSRFProtectionMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(HostCheckMiddleware)


@app.on_event("startup")
def preload_embedding_model():
    """埋め込みモデルを起動時にプリロード（初回検索のレイテンシ回避）"""
    try:
        get_model()
        print("[faiss] Embedding model loaded successfully")
    except Exception as e:
        print(f"[faiss] Warning: Failed to load embedding model: {e}")
        print("[faiss] Similar search will be unavailable")


for router in ALL_ROUTERS:
    app.include_router(router)
