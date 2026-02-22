import re
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from core.templates import templates
from services.twitch import get_user_id

router = APIRouter()

@router.get("/manual", response_class=HTMLResponse)
def manual_page(request: Request):
    return templates.TemplateResponse("manual.html", {"request": request})


@router.get("/add_user", response_class=HTMLResponse)
def add_user_page(request: Request, message: Optional[str] = Query(None), error: Optional[str] = Query(None)):
    try:
        df = pd.read_csv("/host/targetusers.csv")
        users = df.to_dict('records')
    except Exception as e:
        users = []
    return templates.TemplateResponse("add_user.html", {"request": request, "message": message, "error": error, "users": users})


@router.post("/add_user")
def add_user(request: Request, username: str = Form(...)):
    username = username.strip()
    if not username:
        return RedirectResponse(url=str(request.url_for("add_user_page")) + "?error=ユーザ名を入力してください", status_code=303)

    # URLからloginを抽出
    if username.startswith("https://www.twitch.tv/"):
        match = re.match(r"https://www\.twitch\.tv/([^/?]+)", username)
        if match:
            username = match.group(1)
        else:
            return RedirectResponse(url=str(request.url_for("add_user_page")) + "?error=無効なURLです", status_code=303)

    user_id = get_user_id(username)
    if not user_id:
        return RedirectResponse(url=str(request.url_for("add_user_page")) + f"?error=ユーザ {username} が見つかりませんでした", status_code=303)

    try:
        # CSV を読み込み
        df = pd.read_csv("/host/targetusers.csv")
        # すでに存在するかチェック
        if username in df["name"].values:
            return RedirectResponse(url=str(request.url_for("add_user_page")) + f"?error=ユーザ {username} はすでに登録されています", status_code=303)
        # 新しい行を追加
        new_row = pd.DataFrame({"name": [username], "id": [user_id]})
        df = pd.concat([df, new_row], ignore_index=True)
        # CSV に保存
        df.to_csv("/host/targetusers.csv", index=False)
        return RedirectResponse(url=str(request.url_for("add_user_page")) + f"?message=ユーザ {username} (ID: {user_id}) を追加しました", status_code=303)
    except Exception as e:
        import logging
        logging.exception("CSV書き込みエラー")
        return RedirectResponse(url=str(request.url_for("add_user_page")) + "?error=ユーザの追加に失敗しました。しばらく経ってから再度お試しください", status_code=303)
