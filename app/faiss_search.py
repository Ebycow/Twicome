"""
FAISS ベクトル検索モジュール
- SentenceTransformer モデルのシングルトン管理
- ユーザ別 FAISS インデックスの読込・ホットリロード・検索
- 重心距離検索（典型的/珍しい発言）
- 感情アンカー検索（スライダーによる感情ベクトル合成）
"""

import os
import json
import threading
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Optional, List, Tuple, Dict

FAISS_DATA_DIR = os.path.join(os.path.dirname(__file__), "faiss_data")
MODEL_NAME = "hotchpotch/static-embedding-japanese"

# 感情アンカー設定の読み込み
# Docker内では __file__ が /app/faiss_search.py なので同ディレクトリを先に探す
_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_config.json")
if not os.path.exists(_config_path):
    _config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "faiss_config.json")
_emotion_anchors_config: Dict[str, str] = {}
if os.path.exists(_config_path):
    with open(_config_path) as _f:
        _cfg = json.load(_f)
        _emotion_anchors_config = _cfg.get("emotion_anchors", {})

# シングルトン: 埋め込みモデル
_model: Optional[SentenceTransformer] = None
_model_lock = threading.Lock()

# 感情アンカーの埋め込みキャッシュ
_emotion_embeddings: Optional[Dict[str, np.ndarray]] = None
_emotion_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def get_emotion_embeddings() -> Dict[str, np.ndarray]:
    """感情アンカーの埋め込みを取得（初回のみ計算、以降キャッシュ）"""
    global _emotion_embeddings
    if _emotion_embeddings is None:
        with _emotion_lock:
            if _emotion_embeddings is None:
                model = get_model()
                _emotion_embeddings = {}
                for key, anchor_text in _emotion_anchors_config.items():
                    vec = model.encode([anchor_text], normalize_embeddings=True)
                    _emotion_embeddings[key] = np.array(vec[0], dtype=np.float32)
                print(f"[faiss] Emotion anchor embeddings computed: {list(_emotion_embeddings.keys())}")
    return _emotion_embeddings


def get_emotion_axes() -> List[Dict[str, str]]:
    """利用可能な感情軸の一覧を返す（UI表示用）"""
    labels = {
        "joy": "笑い・楽しさ",
        "surprise": "驚き",
        "admiration": "称賛・感動",
        "anger": "怒り・不満",
        "sadness": "悲しみ",
        "cheer": "応援",
    }
    return [{"key": k, "label": labels.get(k, k)} for k in _emotion_anchors_config]


class UserIndex:
    """ユーザ1人分の FAISS インデックスを管理。ファイル変更を検知して自動再読込。"""

    def __init__(self, login: str):
        self.login = login
        self.index: Optional[faiss.Index] = None
        self.comment_ids: List[str] = []
        self.centroid: Optional[np.ndarray] = None
        self.centroid_similarities: Optional[List[float]] = None
        self.last_mtime: float = 0.0
        self.lock = threading.Lock()

    def _index_path(self) -> str:
        return os.path.join(FAISS_DATA_DIR, f"{self.login}.faiss")

    def _meta_path(self) -> str:
        return os.path.join(FAISS_DATA_DIR, f"{self.login}.meta.json")

    def is_available(self) -> bool:
        return os.path.exists(self._index_path()) and os.path.exists(self._meta_path())

    def _check_reload(self):
        """ファイルの mtime を確認し、変更があればインデックスを再読込"""
        index_path = self._index_path()
        if not os.path.exists(index_path):
            return

        try:
            current_mtime = os.path.getmtime(index_path)
        except OSError:
            return

        if current_mtime != self.last_mtime:
            with self.lock:
                try:
                    current_mtime = os.path.getmtime(index_path)
                except OSError:
                    return
                if current_mtime != self.last_mtime:
                    self.index = faiss.read_index(index_path)
                    with open(self._meta_path()) as f:
                        meta = json.load(f)
                    self.comment_ids = meta["comment_ids"]
                    if "centroid" in meta:
                        self.centroid = np.array(meta["centroid"], dtype=np.float32)
                    if "centroid_similarities" in meta:
                        self.centroid_similarities = meta["centroid_similarities"]
                    self.last_mtime = current_mtime
                    print(f"[faiss] Reloaded index for {self.login}: {len(self.comment_ids)} comments")

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
        """類似コメントを検索。[(comment_id, score), ...] を返す。"""
        self._check_reload()
        if self.index is None or len(self.comment_ids) == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.comment_ids):
                continue
            results.append((self.comment_ids[idx], float(score)))
        return results

    def search_by_centroid(self, position: float, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        重心距離でコメントを検索。
        position: 0.0 = 最も典型的（重心に近い）, 1.0 = 最も珍しい（重心から遠い）
        Returns: [(comment_id, centroid_similarity), ...]
        """
        self._check_reload()
        if self.centroid_similarities is None or len(self.comment_ids) == 0:
            return []

        # (index, similarity) のペアを作成してソート
        pairs = list(enumerate(self.centroid_similarities))
        # 類似度が高い順（典型的）→ 低い順（珍しい）
        pairs.sort(key=lambda x: x[1], reverse=True)

        # position に応じてスライス位置を決定
        # 0.0 = 先頭（最も典型的）, 1.0 = 末尾（最も珍しい）
        total = len(pairs)
        max_offset = max(0, total - top_k)
        offset = int(position * max_offset)
        offset = min(offset, max_offset)

        selected = pairs[offset:offset + top_k]
        return [(self.comment_ids[idx], sim) for idx, sim in selected]


# ユーザ別インデックスのレジストリ
_user_indexes: Dict[str, UserIndex] = {}
_registry_lock = threading.Lock()


def get_user_index(login: str) -> UserIndex:
    if login not in _user_indexes:
        with _registry_lock:
            if login not in _user_indexes:
                _user_indexes[login] = UserIndex(login)
    return _user_indexes[login]


def similar_search(login: str, query_text: str, top_k: int = 20) -> List[Tuple[str, float]]:
    """
    高レベルAPI: クエリテキストを埋め込み、ユーザのインデックスから類似コメントを検索。
    Returns: [(comment_id, similarity_score), ...]
    """
    user_index = get_user_index(login)
    if not user_index.is_available():
        return []

    model = get_model()
    query_embedding = model.encode([query_text], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    return user_index.search(query_embedding, top_k)


def centroid_search(login: str, position: float, top_k: int = 50) -> List[Tuple[str, float]]:
    """
    重心距離検索。position: 0.0=典型的, 1.0=珍しい
    Returns: [(comment_id, centroid_similarity), ...]
    """
    user_index = get_user_index(login)
    if not user_index.is_available():
        return []
    return user_index.search_by_centroid(position, top_k)


def emotion_search(login: str, weights: Dict[str, float], top_k: int = 50) -> List[Tuple[str, float]]:
    """
    感情アンカー検索。各感情の重みを合成したベクトルでFAISS検索。
    weights: {"joy": 0.8, "surprise": 0.5, ...}
    Returns: [(comment_id, similarity_score), ...]
    """
    user_index = get_user_index(login)
    if not user_index.is_available():
        return []

    emotion_embs = get_emotion_embeddings()
    if not emotion_embs:
        return []

    # 重みが全部0ならスキップ
    active_weights = {k: v for k, v in weights.items() if v > 0 and k in emotion_embs}
    if not active_weights:
        return []

    # 重み付きベクトル合成
    dim = next(iter(emotion_embs.values())).shape[0]
    combined = np.zeros(dim, dtype=np.float32)
    for key, weight in active_weights.items():
        combined += weight * emotion_embs[key]

    # 正規化
    norm = np.linalg.norm(combined)
    if norm < 1e-8:
        return []
    combined = combined / norm
    combined = combined.reshape(1, -1)

    return user_index.search(combined, top_k)
