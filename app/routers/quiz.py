import random

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import text

from core.config import DEFAULT_PLATFORM
from core.db import SessionLocal
from core.templates import templates
from services.comment_utils import _render_comment_body_html

router = APIRouter()

@router.get("/u/{login}/quiz", response_class=HTMLResponse)
def quiz_page(
    request: Request,
    login: str,
    platform: str = Query(DEFAULT_PLATFORM),
):
    with SessionLocal() as db:
        user_row = db.execute(
            text("""
                SELECT user_id, login, display_name, profile_image_url
                FROM users
                WHERE platform = :platform AND login = :login
                LIMIT 1
            """),
            {"platform": platform, "login": login},
        ).mappings().first()

        if not user_row:
            return templates.TemplateResponse(
                "quiz.html",
                {"request": request, "error": "ユーザが見つかりませんでした", "user": None, "comment_count": 0, "platform": platform},
                status_code=404,
            )

        comment_count = db.execute(
            text("SELECT COUNT(*) AS cnt FROM comments WHERE commenter_user_id = :uid"),
            {"uid": user_row["user_id"]},
        ).mappings().first()["cnt"]

    return templates.TemplateResponse(
        "quiz.html",
        {
            "request": request,
            "error": None,
            "user": dict(user_row),
            "comment_count": comment_count,
            "platform": platform,
        },
    )


@router.get("/api/u/{login}/quiz/start")
def quiz_start_api(
    login: str,
    platform: str = Query(DEFAULT_PLATFORM),
    count: int = Query(30, ge=10, le=100),
):
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

        uid = user_row["user_id"]
        target_count = count // 2
        other_count = count - target_count

        target_comments = db.execute(
            text("""
                SELECT c.body, c.raw_json, c.commenter_login_snapshot, c.commenter_display_name_snapshot,
                       c.user_color, v.title AS vod_title
                FROM comments c
                JOIN vods v ON v.vod_id = c.vod_id
                WHERE c.commenter_user_id = :uid
                  AND CHAR_LENGTH(c.body) >= 3
                ORDER BY RAND()
                LIMIT :lim
            """),
            {"uid": uid, "lim": target_count},
        ).mappings().all()

        other_comments = db.execute(
            text("""
                SELECT c.body, c.raw_json, c.commenter_login_snapshot, c.commenter_display_name_snapshot,
                       c.user_color, v.title AS vod_title
                FROM comments c
                JOIN vods v ON v.vod_id = c.vod_id
                WHERE c.commenter_user_id != :uid
                  AND c.vod_id IN (SELECT DISTINCT vod_id FROM comments WHERE commenter_user_id = :uid)
                  AND CHAR_LENGTH(c.body) >= 3
                ORDER BY RAND()
                LIMIT :lim
            """),
            {"uid": uid, "lim": other_count},
        ).mappings().all()

        questions = []
        for r in target_comments:
            questions.append({
                "body": r["body"],
                "body_html": _render_comment_body_html(r.get("raw_json"), r.get("body")),
                "is_target": True,
                "actual_commenter_display_name": r["commenter_display_name_snapshot"] or r["commenter_login_snapshot"],
                "vod_title": r["vod_title"],
                "user_color": r["user_color"],
            })
        for r in other_comments:
            questions.append({
                "body": r["body"],
                "body_html": _render_comment_body_html(r.get("raw_json"), r.get("body")),
                "is_target": False,
                "actual_commenter_display_name": r["commenter_display_name_snapshot"] or r["commenter_login_snapshot"],
                "vod_title": r["vod_title"],
                "user_color": r["user_color"],
            })

        random.shuffle(questions)

    return {
        "user": dict(user_row),
        "total": len(questions),
        "questions": questions,
    }


