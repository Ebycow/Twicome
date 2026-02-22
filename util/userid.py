#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional

import requests


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


def get_user_id(username: str, client_id: str, access_token: str) -> Optional[str]:
    url = "https://api.twitch.tv/helix/users"
    headers = {
        "Client-ID": client_id,
        "Authorization": f"Bearer {access_token}",
    }

    response = requests.get(url, headers=headers, params={"login": username}, timeout=20)
    response.raise_for_status()
    data = response.json()

    if "data" in data and len(data["data"]) > 0:
        return data["data"][0]["id"]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Get Twitch user ID from username")
    parser.add_argument("username", help="取得したいTwitchチャンネルのユーザー名")
    parser.add_argument(
        "--env-file",
        default=str(Path(__file__).resolve().parents[1] / ".env"),
        help="CLIENT_ID / ACCESS_TOKEN を読む .env のパス",
    )
    args = parser.parse_args()

    env = load_env(Path(args.env_file))
    client_id = env.get("CLIENT_ID", "").strip()
    access_token = env.get("ACCESS_TOKEN", "").strip()

    if not client_id or not access_token:
        raise SystemExit("CLIENT_ID または ACCESS_TOKEN が .env に見つかりません。")

    user_id = get_user_id(args.username, client_id, access_token)
    if user_id:
        print(f"{args.username} の User ID: {user_id}")
        return 0

    print("ユーザーが見つかりませんでした。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
