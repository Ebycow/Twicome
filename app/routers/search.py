from datetime import datetime

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from sqlalchemy import text

from core.config import DEFAULT_PLATFORM
from core.db import SessionLocal
from faiss_search import centroid_search, emotion_search, get_emotion_axes, get_user_index, similar_search
from services.comment_utils import _decorate_comment

router = APIRouter()

@router.get("/api/u/{login}/similar")
def similar_search_api(
    login: str,
    q: str = Query(..., min_length=1),
    platform: str = Query(DEFAULT_PLATFORM),
    top_k: int = Query(20, ge=1, le=100),
):
    """意味的に類似したコメントを検索する"""
    with SessionLocal() as db:
        user_row = db.execute(
            text("""
                SELECT user_id, login, display_name
                FROM users
                WHERE platform = :platform AND login = :login
                LIMIT 1
            """),
            {"platform": platform, "login": login},
        ).mappings().first()
        if not user_row:
            return JSONResponse({"error": "user_not_found"}, status_code=404)

    user_index = get_user_index(login)
    if not user_index.is_available():
        return JSONResponse(
            {"error": "similar_search_not_available", "message": "このユーザの類似検索インデックスはまだ作成されていません"},
            status_code=404,
        )

    results = similar_search(login, q, top_k)

    if not results:
        return {"user": dict(user_row), "query": q, "total": 0, "items": []}

    comment_ids = [r[0] for r in results]
    scores_map = {r[0]: r[1] for r in results}

    placeholders = ",".join([f":id_{i}" for i in range(len(comment_ids))])
    params = {f"id_{i}": cid for i, cid in enumerate(comment_ids)}

    with SessionLocal() as db:
        rows = db.execute(
            text(f"""
                SELECT
                    c.comment_id,
                    c.vod_id,
                    c.offset_seconds,
                    c.comment_created_at_utc,
                    c.commenter_login_snapshot,
                    c.commenter_display_name_snapshot,
                    c.body,
                    c.raw_json,
                    c.user_color,
                    c.bits_spent,
                    c.twicome_likes_count,
                    c.twicome_dislikes_count,
                    cn.note AS community_note_body,
                    cn.eligible AS cn_eligible,
                    cn.status AS cn_status,
                    cn.verifiability AS cn_verifiability,
                    cn.harm_risk AS cn_harm_risk,
                    cn.exaggeration AS cn_exaggeration,
                    cn.evidence_gap AS cn_evidence_gap,
                    cn.subjectivity AS cn_subjectivity,
                    cn.issues AS cn_issues,
                    cn.ask AS cn_ask,
                    v.title AS vod_title,
                    v.url AS vod_url,
                    v.youtube_url AS youtube_url,
                    v.created_at_utc AS vod_created_at_utc,
                    u.login AS owner_login,
                    u.display_name AS owner_display_name
                FROM comments c
                JOIN vods v ON v.vod_id = c.vod_id
                JOIN users u ON u.user_id = v.owner_user_id
                LEFT JOIN community_notes cn ON cn.comment_id = c.comment_id
                WHERE c.comment_id IN ({placeholders})
            """),
            params,
        ).mappings().all()

    rows_map = {r["comment_id"]: r for r in rows}
    comments = []
    now = datetime.utcnow()
    for cid, score in results:
        r = rows_map.get(cid)
        if not r:
            continue
        comment = _decorate_comment(r, now)
        comment["similarity_score"] = round(score, 4)
        comments.append(comment)

    return {"user": dict(user_row), "query": q, "total": len(comments), "items": comments}


def _fetch_comment_details(results: list) -> list:
    """FAISS検索結果 [(comment_id, score), ...] からDB詳細を取得してコメント一覧を構築"""
    if not results:
        return []

    comment_ids = [r[0] for r in results]
    placeholders = ",".join([f":id_{i}" for i in range(len(comment_ids))])
    params = {f"id_{i}": cid for i, cid in enumerate(comment_ids)}

    with SessionLocal() as db:
        rows = db.execute(
            text(f"""
                SELECT
                    c.comment_id, c.vod_id, c.offset_seconds, c.comment_created_at_utc,
                    c.commenter_login_snapshot, c.commenter_display_name_snapshot,
                    c.body, c.raw_json, c.user_color, c.bits_spent,
                    c.twicome_likes_count, c.twicome_dislikes_count,
                    cn.note AS community_note_body,
                    cn.eligible AS cn_eligible,
                    cn.status AS cn_status,
                    cn.verifiability AS cn_verifiability,
                    cn.harm_risk AS cn_harm_risk,
                    cn.exaggeration AS cn_exaggeration,
                    cn.evidence_gap AS cn_evidence_gap,
                    cn.subjectivity AS cn_subjectivity,
                    cn.issues AS cn_issues,
                    cn.ask AS cn_ask,
                    v.title AS vod_title, v.url AS vod_url, v.youtube_url AS youtube_url, v.created_at_utc AS vod_created_at_utc,
                    u.login AS owner_login, u.display_name AS owner_display_name
                FROM comments c
                JOIN vods v ON v.vod_id = c.vod_id
                JOIN users u ON u.user_id = v.owner_user_id
                LEFT JOIN community_notes cn ON cn.comment_id = c.comment_id
                WHERE c.comment_id IN ({placeholders})
            """),
            params,
        ).mappings().all()

    rows_map = {r["comment_id"]: r for r in rows}
    comments = []
    now = datetime.utcnow()
    for cid, score in results:
        r = rows_map.get(cid)
        if not r:
            continue
        comment = _decorate_comment(r, now)
        comment["similarity_score"] = round(score, 4)
        comments.append(comment)
    return comments


@router.get("/api/u/{login}/centroid")
def centroid_search_api(
    login: str,
    position: float = Query(0.5, ge=0.0, le=1.0),
    platform: str = Query(DEFAULT_PLATFORM),
    top_k: int = Query(50, ge=1, le=100),
):
    """重心距離でコメントを検索。position: 0.0=典型的, 1.0=珍しい"""
    with SessionLocal() as db:
        user_row = db.execute(
            text("SELECT user_id, login, display_name FROM users WHERE platform = :platform AND login = :login LIMIT 1"),
            {"platform": platform, "login": login},
        ).mappings().first()
        if not user_row:
            return JSONResponse({"error": "user_not_found"}, status_code=404)

    user_index = get_user_index(login)
    if not user_index.is_available():
        return JSONResponse({"error": "index_not_available"}, status_code=404)

    results = centroid_search(login, position, top_k)
    comments = _fetch_comment_details(results)
    return {"user": dict(user_row), "position": position, "total": len(comments), "items": comments}


@router.get("/api/u/{login}/emotion")
def emotion_search_api(
    login: str,
    platform: str = Query(DEFAULT_PLATFORM),
    top_k: int = Query(50, ge=1, le=100),
    joy: float = Query(0.0, ge=0.0, le=1.0),
    surprise: float = Query(0.0, ge=0.0, le=1.0),
    admiration: float = Query(0.0, ge=0.0, le=1.0),
    anger: float = Query(0.0, ge=0.0, le=1.0),
    sadness: float = Query(0.0, ge=0.0, le=1.0),
    cheer: float = Query(0.0, ge=0.0, le=1.0),
):
    """感情スライダーでコメントを検索"""
    with SessionLocal() as db:
        user_row = db.execute(
            text("SELECT user_id, login, display_name FROM users WHERE platform = :platform AND login = :login LIMIT 1"),
            {"platform": platform, "login": login},
        ).mappings().first()
        if not user_row:
            return JSONResponse({"error": "user_not_found"}, status_code=404)

    user_index = get_user_index(login)
    if not user_index.is_available():
        return JSONResponse({"error": "index_not_available"}, status_code=404)

    weights = {"joy": joy, "surprise": surprise, "admiration": admiration, "anger": anger, "sadness": sadness, "cheer": cheer}
    if all(v == 0 for v in weights.values()):
        return {"user": dict(user_row), "weights": weights, "total": 0, "items": []}

    results = emotion_search(login, weights, top_k)
    comments = _fetch_comment_details(results)
    return {"user": dict(user_row), "weights": weights, "total": len(comments), "items": comments}


@router.get("/api/emotion_axes")
def emotion_axes_api():
    """利用可能な感情軸の一覧"""
    return {"axes": get_emotion_axes()}


