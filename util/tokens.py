#!/usr/bin/env python3

import argparse
from pathlib import Path

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Get Twitch access token with client credentials")
    parser.add_argument(
        "--env-file",
        default=str(Path(__file__).resolve().parents[1] / ".env"),
        help="CLIENT_ID / CLIENT_SECRET を読む .env のパス",
    )
    parser.add_argument(
        "--scope",
        default="user:read:broadcast",
        help="要求するスコープ (default: user:read:broadcast)",
    )
    args = parser.parse_args()

    env = load_env(Path(args.env_file))
    client_id = env.get("CLIENT_ID", "").strip()
    client_secret = env.get("CLIENT_SECRET", "").strip()

    if not client_id or not client_secret:
        raise SystemExit("CLIENT_ID または CLIENT_SECRET が .env に見つかりません。")

    response = requests.post(
        "https://id.twitch.tv/oauth2/token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
            "scope": args.scope,
        },
        timeout=20,
    )
    response.raise_for_status()
    response_data = response.json()

    if "access_token" in response_data:
        print(f"取得した ACCESS_TOKEN: {response_data['access_token']}")
        return 0

    print("アクセストークンの取得に失敗しました:", response_data)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
