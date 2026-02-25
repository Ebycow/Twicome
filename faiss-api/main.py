"""
FAISS API サービス

埋め込みモデルとFAISSインデックス管理を一元化するサービス。
app と batch からHTTP経由で利用する。
"""

import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --- 設定 ---
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "hotchpotch/static-embedding-japanese")
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "512"))
CONFIG_PATH = Path(os.getenv("FAISS_CONFIG_PATH", "/app/faiss_config.json"))

# faiss_config.json から感情アンカーを読み込む
_emotion_anchors_config: Dict[str, str] = {}
if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as _f:
        _cfg = json.load(_f)
        _emotion_anchors_config = _cfg.get("emotion_anchors", {})

app = FastAPI(title="FAISS API", version="1.0.0")

# --- シングルトン: 埋め込みモデル ---
_model: Optional[SentenceTransformer] = None
_model_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


# --- 感情アンカーの埋め込みキャッシュ ---
_emotion_embeddings: Optional[Dict[str, np.ndarray]] = None
_emotion_lock = threading.Lock()


def get_emotion_embeddings() -> Dict[str, np.ndarray]:
    global _emotion_embeddings
    if _emotion_embeddings is None:
        with _emotion_lock:
            if _emotion_embeddings is None:
                model = get_model()
                _emotion_embeddings = {}
                for key, anchor_text in _emotion_anchors_config.items():
                    vec = model.encode([anchor_text], normalize_embeddings=True)
                    _emotion_embeddings[key] = np.array(vec[0], dtype=np.float32)
                print(f"[faiss-api] 感情アンカー埋め込み計算完了: {list(_emotion_embeddings.keys())}")
    return _emotion_embeddings


# --- ユーザ別インデックス管理 ---
class UserIndex:
    """ユーザ1人分のFAISSインデックスを管理する"""

    def __init__(self, login: str):
        self.login = login
        self.index: Optional[faiss.Index] = None
        self.comment_ids: List[str] = []
        self.centroid: Optional[np.ndarray] = None
        self.centroid_similarities: Optional[List[float]] = None
        self.lock = threading.Lock()

    @property
    def index_path(self) -> Path:
        return DATA_DIR / f"{self.login}.faiss"

    @property
    def meta_path(self) -> Path:
        return DATA_DIR / f"{self.login}.meta.json"

    def is_available(self) -> bool:
        return self.index_path.exists() and self.meta_path.exists()

    def load(self):
        """ディスクからインデックスを読み込む（ロック内から呼ぶこと）"""
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path) as f:
            meta = json.load(f)
        self.comment_ids = meta["comment_ids"]
        self.centroid = np.array(meta["centroid"], dtype=np.float32) if "centroid" in meta else None
        self.centroid_similarities = meta.get("centroid_similarities")
        print(f"[faiss-api] インデックス読み込み [{self.login}]: {len(self.comment_ids)} 件")

    def ensure_loaded(self):
        """インデックスが未ロードなら読み込む"""
        with self.lock:
            if self.index is None and self.is_available():
                self.load()

    def update(self, new_ids: List[str], new_embeddings: np.ndarray) -> int:
        """新規埋め込みをインデックスに追加し、重心を再計算してアトミック保存する。追加件数を返す"""
        with self.lock:
            dim = new_embeddings.shape[1]

            if self.index is None:
                if self.is_available():
                    self.load()
                else:
                    self.index = faiss.IndexFlatIP(dim)
                    self.comment_ids = []

            # ロック内で再チェック（並行リクエスト対策）
            existing_ids = set(self.comment_ids)
            mask = [i for i, cid in enumerate(new_ids) if cid not in existing_ids]
            if not mask:
                return 0
            new_ids = [new_ids[i] for i in mask]
            new_embeddings = new_embeddings[mask]

            self.index.add(new_embeddings)
            self.comment_ids.extend(new_ids)

            # 重心と各コメントのコサイン類似度を再計算
            n_total = self.index.ntotal
            all_vectors = np.zeros((n_total, dim), dtype=np.float32)
            for i in range(n_total):
                all_vectors[i] = self.index.reconstruct(i)
            centroid = np.mean(all_vectors, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm
            self.centroid = centroid
            self.centroid_similarities = (all_vectors @ centroid).tolist()

            # アトミック書き込み
            tmp_index = str(self.index_path) + ".tmp"
            tmp_meta = str(self.meta_path) + ".tmp"
            faiss.write_index(self.index, tmp_index)
            meta = {
                "comment_ids": self.comment_ids,
                "total_comments": len(self.comment_ids),
                "embedding_dim": dim,
                "centroid": self.centroid.tolist(),
                "centroid_similarities": self.centroid_similarities,
            }
            with open(tmp_meta, "w") as f:
                json.dump(meta, f, ensure_ascii=False)
            os.replace(tmp_index, str(self.index_path))
            os.replace(tmp_meta, str(self.meta_path))

            return len(new_ids)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """類似検索。[(comment_id, score), ...] を返す"""
        k = min(top_k, self.index.ntotal)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.comment_ids):
                results.append((self.comment_ids[idx], float(score)))
        return results

    def search_centroid(self, position: float, top_k: int) -> List[Tuple[str, float]]:
        """重心距離検索。position=0.0→典型的, 1.0→珍しい"""
        if self.centroid_similarities is None:
            return []
        pairs = list(enumerate(self.centroid_similarities))
        pairs.sort(key=lambda x: x[1], reverse=True)
        total = len(pairs)
        max_offset = max(0, total - top_k)
        offset = min(int(position * max_offset), max_offset)
        selected = pairs[offset:offset + top_k]
        return [(self.comment_ids[idx], sim) for idx, sim in selected]


# ユーザ別インデックスのレジストリ
_user_indexes: Dict[str, UserIndex] = {}
_registry_lock = threading.Lock()


def get_user_index(login: str) -> UserIndex:
    if login not in _user_indexes:
        with _registry_lock:
            if login not in _user_indexes:
                ui = UserIndex(login)
                if ui.is_available():
                    ui.load()
                _user_indexes[login] = ui
    return _user_indexes[login]


# --- リクエスト/レスポンスモデル ---
class EmbedRequest(BaseModel):
    texts: List[str]
    normalize: bool = True


class IndexUpdateRequest(BaseModel):
    comment_ids: List[str]
    texts: List[str]


class SimilarSearchRequest(BaseModel):
    query: str
    top_k: int = 20


class CentroidSearchRequest(BaseModel):
    position: float = 0.5
    top_k: int = 50


class EmotionSearchRequest(BaseModel):
    weights: Dict[str, float]
    top_k: int = 50


# --- 起動時処理 ---
@app.on_event("startup")
def startup():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[faiss-api] モデル読み込み中: {MODEL_NAME}")
    get_model()
    if _emotion_anchors_config:
        get_emotion_embeddings()
    print(f"[faiss-api] 起動完了。データディレクトリ: {DATA_DIR}")


# --- エンドポイント ---
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/embed")
def embed(req: EmbedRequest):
    if not req.texts:
        return {"embeddings": []}
    model = get_model()
    embeddings = model.encode(req.texts, normalize_embeddings=req.normalize)
    return {"embeddings": embeddings.tolist()}


@app.get("/emotion_axes")
def emotion_axes():
    labels = {
        "joy": "笑い・楽しさ",
        "surprise": "驚き",
        "admiration": "称賛・感動",
        "anger": "怒り・不満",
        "sadness": "悲しみ",
        "cheer": "応援",
    }
    axes = [{"key": k, "label": labels.get(k, k)} for k in _emotion_anchors_config]
    return {"axes": axes}


@app.get("/index/{login}/status")
def index_status(login: str):
    ui = get_user_index(login)
    if not ui.is_available():
        raise HTTPException(status_code=404, detail="index_not_available")
    return {"login": login, "total": len(ui.comment_ids)}


@app.post("/index/update/{login}")
def update_index(login: str, req: IndexUpdateRequest):
    if len(req.comment_ids) != len(req.texts):
        raise HTTPException(status_code=400, detail="comment_ids と texts の件数が一致しません")

    ui = get_user_index(login)

    model = get_model()
    embeddings = model.encode(
        req.texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    added = ui.update(req.comment_ids, embeddings)
    print(f"[faiss-api] インデックス更新 [{login}]: +{added} 件, 合計 {len(ui.comment_ids)} 件")

    return {"status": "ok", "added": added, "total": len(ui.comment_ids), "login": login}


@app.post("/search/similar/{login}")
def search_similar(login: str, req: SimilarSearchRequest):
    ui = get_user_index(login)
    if not ui.is_available() or ui.index is None:
        raise HTTPException(status_code=404, detail="index_not_available")

    model = get_model()
    query_emb = model.encode([req.query], normalize_embeddings=True)
    query_emb = np.array(query_emb, dtype=np.float32)

    with ui.lock:
        results = ui.search(query_emb, req.top_k)

    return {"results": [{"comment_id": cid, "score": score} for cid, score in results]}


@app.post("/search/centroid/{login}")
def search_centroid(login: str, req: CentroidSearchRequest):
    ui = get_user_index(login)
    if not ui.is_available() or ui.centroid_similarities is None:
        raise HTTPException(status_code=404, detail="index_not_available")

    with ui.lock:
        results = ui.search_centroid(req.position, req.top_k)

    return {"results": [{"comment_id": cid, "score": score} for cid, score in results]}


@app.post("/search/emotion/{login}")
def search_emotion(login: str, req: EmotionSearchRequest):
    ui = get_user_index(login)
    if not ui.is_available() or ui.index is None:
        raise HTTPException(status_code=404, detail="index_not_available")

    emotion_embs = get_emotion_embeddings()
    active_weights = {k: v for k, v in req.weights.items() if v > 0 and k in emotion_embs}
    if not active_weights:
        raise HTTPException(status_code=400, detail="有効な感情ウェイトが設定されていません")

    dim = next(iter(emotion_embs.values())).shape[0]
    combined = np.zeros(dim, dtype=np.float32)
    for key, weight in active_weights.items():
        combined += weight * emotion_embs[key]

    norm = np.linalg.norm(combined)
    if norm < 1e-8:
        raise HTTPException(status_code=400, detail="合成ベクトルがゼロです")
    combined = (combined / norm).reshape(1, -1)

    with ui.lock:
        results = ui.search(combined, req.top_k)

    return {"results": [{"comment_id": cid, "score": score} for cid, score in results]}
