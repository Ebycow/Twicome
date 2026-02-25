"""
FAISS 検索クライアント

faiss-api サービスへのHTTP通信を担うクライアントモジュール。
FAISS_API_URL が未設定の場合はすべての関数が空/Falseを返す。
"""

import os
from typing import Dict, List, Optional, Tuple

import requests

FAISS_API_URL: str = os.getenv("FAISS_API_URL", "").strip().rstrip("/")
_REQUEST_TIMEOUT = 30


def _is_enabled() -> bool:
    return bool(FAISS_API_URL)


def ping_faiss_api() -> bool:
    """faiss-api の死活確認。接続できれば True を返す"""
    if not _is_enabled():
        return False
    try:
        resp = requests.get(f"{FAISS_API_URL}/health", timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        raise RuntimeError(f"faiss-api ({FAISS_API_URL}) に接続できません: {e}") from e


def is_index_available(login: str) -> bool:
    """指定ユーザのFAISSインデックスが利用可能かチェックする"""
    if not _is_enabled():
        return False
    try:
        resp = requests.get(f"{FAISS_API_URL}/index/{login}/status", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def get_emotion_axes() -> List[Dict[str, str]]:
    """利用可能な感情軸の一覧を返す（UI表示用）"""
    if not _is_enabled():
        return []
    try:
        resp = requests.get(f"{FAISS_API_URL}/emotion_axes", timeout=10)
        resp.raise_for_status()
        return resp.json().get("axes", [])
    except Exception:
        return []


def similar_search(login: str, query_text: str, top_k: int = 20) -> Optional[List[Tuple[str, float]]]:
    """
    意味的類似検索。
    Returns: [(comment_id, score), ...] または None (インデックス未作成)
    """
    if not _is_enabled():
        return None
    resp = requests.post(
        f"{FAISS_API_URL}/search/similar/{login}",
        json={"query": query_text, "top_k": top_k},
        timeout=_REQUEST_TIMEOUT,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return [(r["comment_id"], r["score"]) for r in resp.json()["results"]]


def centroid_search(login: str, position: float, top_k: int = 50) -> Optional[List[Tuple[str, float]]]:
    """
    重心距離検索。position: 0.0=典型的, 1.0=珍しい
    Returns: [(comment_id, centroid_similarity), ...] または None
    """
    if not _is_enabled():
        return None
    resp = requests.post(
        f"{FAISS_API_URL}/search/centroid/{login}",
        json={"position": position, "top_k": top_k},
        timeout=_REQUEST_TIMEOUT,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return [(r["comment_id"], r["score"]) for r in resp.json()["results"]]


def emotion_search(login: str, weights: Dict[str, float], top_k: int = 50) -> Optional[List[Tuple[str, float]]]:
    """
    感情アンカー検索。各感情の重みを合成したベクトルで検索。
    weights: {"joy": 0.8, "surprise": 0.5, ...}
    Returns: [(comment_id, score), ...] または None
    """
    if not _is_enabled():
        return None
    resp = requests.post(
        f"{FAISS_API_URL}/search/emotion/{login}",
        json={"weights": weights, "top_k": top_k},
        timeout=_REQUEST_TIMEOUT,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return [(r["comment_id"], r["score"]) for r in resp.json()["results"]]
