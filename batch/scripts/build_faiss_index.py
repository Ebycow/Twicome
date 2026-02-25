"""
FAISS インデックス構築スクリプト (faiss-api クライアント版)

faiss_config.json に記載されたユーザのコメントを MySQL から取得し、
faiss-api へ送信することでインデックスを更新する。
埋め込み生成・FAISS操作はすべて faiss-api 側で行う。

Usage:
    python build_faiss_index.py                    # 全ユーザ
    python build_faiss_index.py username           # 特定ユーザのみ
"""

import json
import os
import sys
from pathlib import Path

import mysql.connector
import requests
from dotenv import load_dotenv

# -----------------------------------------------
# 設定
# -----------------------------------------------
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
ENV_PATH = Path(os.getenv("ENV_FILE", str(PROJECT_ROOT / ".env")))
if not ENV_PATH.is_absolute():
    ENV_PATH = PROJECT_ROOT / ENV_PATH
load_dotenv(str(ENV_PATH))

FAISS_API_URL = os.getenv("FAISS_API_URL", "").strip().rstrip("/")
if not FAISS_API_URL:
    print("FAISS_API_URL が設定されていません。FAISSインデックス構築をスキップします。")
    sys.exit(0)

CONFIG_PATH = Path(os.getenv("FAISS_CONFIG_PATH", PROJECT_ROOT / "faiss_config.json"))
with open(CONFIG_PATH) as f:
    config = json.load(f)

MYSQL_HOST = os.getenv("MYSQL_HOST", "db")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "appuser")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
if not MYSQL_PASSWORD:
    raise RuntimeError("MYSQL_PASSWORD が設定されていません。")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "appdb")

# チャンクサイズ: 大規模データを分割して送信
CHUNK_SIZE = 1000


def update_index_for_user(conn, login: str):
    """1ユーザのインデックスを faiss-api 経由で更新する"""
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """
        SELECT c.comment_id, c.body
        FROM comments c
        JOIN users u ON u.user_id = c.commenter_user_id
        WHERE u.login = %s AND u.platform = 'twitch'
          AND c.body IS NOT NULL AND c.body != ''
        ORDER BY c.comment_id
        """,
        (login,),
    )
    rows = cur.fetchall()
    cur.close()

    if not rows:
        print(f"  コメントなし、スキップ")
        return

    print(f"  DB上の全コメント: {len(rows)} 件 → faiss-api へ送信中...")

    total_added = 0
    for i in range(0, len(rows), CHUNK_SIZE):
        chunk = rows[i:i + CHUNK_SIZE]
        chunk_ids = [r["comment_id"] for r in chunk]
        chunk_texts = [r["body"] for r in chunk]

        resp = requests.post(
            f"{FAISS_API_URL}/index/update/{login}",
            json={"comment_ids": chunk_ids, "texts": chunk_texts},
            timeout=180,
        )
        if not resp.ok:
            print(f"  faiss-api エラー [{resp.status_code}]: {resp.text}")
        resp.raise_for_status()
        result = resp.json()
        total_added += result["added"]

        chunk_num = i // CHUNK_SIZE + 1
        total_chunks = (len(rows) + CHUNK_SIZE - 1) // CHUNK_SIZE
        if result["added"] > 0:
            print(f"  チャンク {chunk_num}/{total_chunks}: +{result['added']} 件 (合計 {result['total']} 件)")

    print(f"  完了: 新規追加 {total_added} 件")


def main():
    # 対象ユーザの決定
    if len(sys.argv) > 1:
        target_users = sys.argv[1:]
    else:
        target_users = config["indexed_users"]

    print(f"対象ユーザ: {target_users}")
    print(f"faiss-api URL: {FAISS_API_URL}")

    # faiss-api の死活確認
    try:
        health = requests.get(f"{FAISS_API_URL}/health", timeout=10)
        health.raise_for_status()
        print(f"faiss-api 接続確認: {health.json()}")
    except Exception as e:
        print(f"faiss-api に接続できません: {e}")
        sys.exit(1)

    print("MySQL 接続中...")
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
    )

    try:
        for login in target_users:
            print(f"\n[{login}]")
            update_index_for_user(conn, login)
    finally:
        conn.close()

    print("\n全て完了。")


if __name__ == "__main__":
    main()
