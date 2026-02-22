"""
FAISS インデックス構築スクリプト

faiss_config.json に記載されたユーザのコメントを MySQL から取得し、
hotchpotch/static-embedding-japanese で埋め込みを生成して FAISS に保存する。
増分更新対応: 既にインデックス済みのコメントはスキップする。

Usage:
    python build_faiss_index.py                    # 全ユーザ
    python build_faiss_index.py username   # 特定ユーザのみ
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import faiss
import mysql.connector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# -----------------------------------------------
# 設定
# -----------------------------------------------
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
ENV_PATH = Path(os.getenv("ENV_FILE", str(PROJECT_ROOT / ".env")))
if not ENV_PATH.is_absolute():
    ENV_PATH = PROJECT_ROOT / ENV_PATH
load_dotenv(str(ENV_PATH))
CONFIG_PATH = Path(os.getenv("FAISS_CONFIG_PATH", PROJECT_ROOT / "faiss_config.json"))

with open(CONFIG_PATH) as f:
    config = json.load(f)

MODEL_NAME = config["embedding_model"]
BATCH_SIZE = config["batch_size"]
faiss_data_dir = Path(config["faiss_data_dir"])
if not faiss_data_dir.is_absolute():
    faiss_data_dir = PROJECT_ROOT / faiss_data_dir
FAISS_DATA_DIR = str(faiss_data_dir)

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "appuser")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
if not MYSQL_PASSWORD:
    raise RuntimeError("MYSQL_PASSWORD is not set. Set MYSQL_PASSWORD in .env or environment variables.")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "appdb")


def build_index_for_user(model, conn, login: str):
    """1ユーザ分のインデックスを構築（増分更新対応）"""
    os.makedirs(FAISS_DATA_DIR, exist_ok=True)

    index_path = os.path.join(FAISS_DATA_DIR, f"{login}.faiss")
    meta_path = os.path.join(FAISS_DATA_DIR, f"{login}.meta.json")

    # 既存メタデータの読込
    existing_ids = set()
    existing_id_list = []
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        existing_id_list = meta["comment_ids"]
        existing_ids = set(existing_id_list)
        print(f"  既存インデックス: {len(existing_id_list)} 件")

    # MySQL からユーザの全コメントを取得
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT c.comment_id, c.body
        FROM comments c
        JOIN users u ON u.user_id = c.commenter_user_id
        WHERE u.login = %s AND u.platform = 'twitch'
        ORDER BY c.comment_id
    """, (login,))
    rows = cur.fetchall()
    cur.close()

    print(f"  DB上の全コメント: {len(rows)} 件")

    # 新規コメントのみ抽出
    new_rows = [r for r in rows if r["comment_id"] not in existing_ids]

    if not new_rows:
        print(f"  新規コメントなし、スキップ")
        return

    print(f"  新規コメント: {len(new_rows)} 件 → 埋め込み生成中...")

    # 埋め込み生成
    bodies = [r["body"] for r in new_rows]
    new_ids = [r["comment_id"] for r in new_rows]
    embeddings = model.encode(
        bodies,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]

    # FAISS インデックスの読込 or 新規作成
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = faiss.IndexFlatIP(dim)

    # ベクトル追加
    index.add(embeddings)

    # メタデータ更新
    updated_id_list = existing_id_list + new_ids

    # 重心（centroid）と各コメントのコサイン類似度を計算
    print(f"  重心を計算中...")
    n_total = index.ntotal
    all_vectors = np.zeros((n_total, dim), dtype=np.float32)
    for i in range(n_total):
        all_vectors[i] = index.reconstruct(i)
    centroid = np.mean(all_vectors, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm
    # コサイン類似度 = dot(comment, centroid)（両方正規化済み）
    centroid_similarities = (all_vectors @ centroid).tolist()

    meta = {
        "comment_ids": updated_id_list,
        "last_indexed_at": datetime.now(timezone.utc).isoformat(),
        "total_comments": len(updated_id_list),
        "embedding_dim": dim,
        "centroid": centroid.tolist(),
        "centroid_similarities": centroid_similarities,
    }

    # アトミック書き込み（.tmp に書いてから rename）
    tmp_index = index_path + ".tmp"
    tmp_meta = meta_path + ".tmp"
    faiss.write_index(index, tmp_index)
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, ensure_ascii=False)
    os.replace(tmp_index, index_path)
    os.replace(tmp_meta, meta_path)

    print(f"  完了: 合計 {len(updated_id_list)} 件 (新規 {len(new_ids)} 件)")


def main():
    # 対象ユーザの決定
    if len(sys.argv) > 1:
        target_users = sys.argv[1:]
    else:
        target_users = config["indexed_users"]

    print(f"対象ユーザ: {target_users}")
    print(f"埋め込みモデル: {MODEL_NAME}")

    print("モデル読み込み中...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

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
            build_index_for_user(model, conn, login)
    finally:
        conn.close()

    print("\n全て完了。")


if __name__ == "__main__":
    main()
