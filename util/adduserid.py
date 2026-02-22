import requests
import pandas as pd
import sys
from pathlib import Path


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env

    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, value = s.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        env[key] = value
    return env


ENV = load_env(Path(__file__).resolve().parents[1] / ".env")
CLIENT_ID = ENV.get("CLIENT_ID", "").strip()
ACCESS_TOKEN = ENV.get("ACCESS_TOKEN", "").strip()

def get_user_id(username):
    url = f"https://api.twitch.tv/helix/users?login={username}"
    headers = {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    if "data" in data and len(data["data"]) > 0:
        user_id = data["data"][0]["id"]
        return user_id
    else:
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python util/adduserid.py <username>")
        sys.exit(1)

    if not CLIENT_ID or not ACCESS_TOKEN:
        print("CLIENT_ID または ACCESS_TOKEN が .env に見つかりません。")
        sys.exit(1)

    username = sys.argv[1]
    user_id = get_user_id(username)

    if user_id:
        # CSV を読み込み
        # TODO: 現状固定なので環境変数から読む必要あり
        df = pd.read_csv("targetusers.csv")
        # 新しい行を追加
        new_row = pd.DataFrame({"name": [username], " id": [user_id]})
        df = pd.concat([df, new_row], ignore_index=True)
        # CSV に保存
        df.to_csv("targetusers.csv", index=False)
        print(f"Added {username} (ID: {user_id}) to targetusers.csv")
    else:
        print(f"User {username} not found.")
