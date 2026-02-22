#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import tempfile
import fcntl
from typing import Dict, List, Tuple

import requests


def parse_env_lines(lines: List[str]) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """
    Returns:
      - kv: parsed key/value map (only for KEY=VALUE lines)
      - parsed: list of tuples (kind, content) where kind in {"raw", "kv"} for reconstruction
    """
    kv: Dict[str, str] = {}
    parsed: List[Tuple[str, str]] = []

    for line in lines:
        s = line.rstrip("\n")
        stripped = s.strip()

        # Keep comments/blank lines as-is
        if stripped == "" or stripped.startswith("#"):
            parsed.append(("raw", s))
            continue

        if "=" not in s:
            parsed.append(("raw", s))
            continue

        key, val = s.split("=", 1)
        key = key.strip()
        val = val.strip()

        # Remove optional surrounding quotes for internal representation
        if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
            val_unquoted = val[1:-1]
        else:
            val_unquoted = val

        kv[key] = val_unquoted
        parsed.append(("kv", key))

    return kv, parsed


def render_env(lines_parsed: List[Tuple[str, str]], existing_kv: Dict[str, str], updates: Dict[str, str]) -> List[str]:
    """
    Reconstruct env file lines, updating keys in-place, preserving comments and unknown raw lines.
    """
    merged = dict(existing_kv)
    merged.update(updates)

    seen_keys = set()
    out: List[str] = []

    for kind, content in lines_parsed:
        if kind == "raw":
            out.append(content)
            continue

        # kind == "kv" and content is key
        key = content
        seen_keys.add(key)
        val = merged.get(key, "")

        # Quote safely (simple approach)
        val_escaped = val.replace('"', r'\"')
        out.append(f'{key}="{val_escaped}"')

    # Append any new keys not previously present
    for key, val in merged.items():
        if key in seen_keys:
            continue
        val_escaped = val.replace('"', r'\"')
        out.append(f'{key}="{val_escaped}"')

    # Ensure trailing newline
    return [l + "\n" for l in out]


def atomic_write(path: str, content_lines: List[str], mode: int = 0o600) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmppath = tempfile.mkstemp(prefix=".tokenenv.", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(content_lines)
            f.flush()
            os.fsync(f.fileno())

        os.chmod(tmppath, mode)
        os.replace(tmppath, path)  # atomic on same filesystem
    finally:
        try:
            if os.path.exists(tmppath):
                os.remove(tmppath)
        except OSError:
            pass


def refresh_token(client_id: str, client_secret: str, scope: str) -> Dict[str, str]:
    url = "https://id.twitch.tv/oauth2/token"
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
    }
    if scope.strip():
        data["scope"] = scope.strip()

    r = requests.post(url, data=data, timeout=20)
    r.raise_for_status()
    j = r.json()

    if "access_token" not in j or "expires_in" not in j:
        raise RuntimeError(f"Unexpected token response: {j}")

    now = int(time.time())
    expires_at = now + int(j["expires_in"])

    return {
        "ACCESS_TOKEN": str(j["access_token"]),
        "EXPIRES_AT": str(expires_at),
        "TOKEN_TYPE": str(j.get("token_type", "")),
        "SCOPE": scope.strip(),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Refresh Twitch access token and update .env")
    parser.add_argument("env_path", help="Path to .env (will be created/updated)")
    parser.add_argument("--buffer-seconds", type=int, default=int(os.getenv("TOKEN_REFRESH_BUFFER", "300")),
                        help="Refresh when expires_at is within this buffer (default: 300)")
    parser.add_argument("--scope", default=os.getenv("TWITCH_SCOPE", "user:read:broadcast"),
                        help="Scope for client_credentials (default: env TWITCH_SCOPE or user:read:broadcast)")
    parser.add_argument("--client-id", default=None,
                        help="Twitch Client ID (or set TWITCH_CLIENT_ID env var, or CLIENT_ID in .env)")
    parser.add_argument("--client-secret", default=None,
                        help="Twitch Client Secret (or set TWITCH_CLIENT_SECRET env var, or CLIENT_SECRET in .env)")
    args = parser.parse_args()

    # クライアント認証情報の読み込み優先順位:
    # 1. コマンドライン引数
    # 2. 環境変数 (TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET)
    # 3. .envファイル内の CLIENT_ID / CLIENT_SECRET

    # まず .env から読み込み試行
    env_client_id = None
    env_client_secret = None
    if os.path.exists(args.env_path):
        with open(args.env_path, "r", encoding="utf-8") as f:
            env_kv, _ = parse_env_lines(f.readlines())
            env_client_id = env_kv.get("CLIENT_ID", "").strip() or None
            env_client_secret = env_kv.get("CLIENT_SECRET", "").strip() or None

    client_id = (
        args.client_id
        or os.getenv("TWITCH_CLIENT_ID")
        or env_client_id
    )
    client_secret = (
        args.client_secret
        or os.getenv("TWITCH_CLIENT_SECRET")
        or env_client_secret
    )

    if not client_id or not client_secret:
        raise SystemExit(
            "CLIENT_ID/CLIENT_SECRET が見つかりません。以下のいずれかで設定してください:\n"
            "  1. コマンドライン引数: --client-id, --client-secret\n"
            "  2. 環境変数: TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET\n"
            "  3. .envファイル内: CLIENT_ID, CLIENT_SECRET"
        )

    path = args.env_path
    buffer_seconds = args.buffer_seconds
    scope = args.scope

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    # Use a lock file to avoid concurrent refresh races
    lock_path = path + ".lock"
    with open(lock_path, "a+", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)

        # Read current .env if exists
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = ["# Auto-managed by refresh_twitch_token.py\n"]

        kv, parsed = parse_env_lines(lines)

        now = int(time.time())
        expires_at_str = kv.get("EXPIRES_AT", "").strip()
        current_token = kv.get("ACCESS_TOKEN", "").strip()

        should_refresh = True
        if current_token and expires_at_str.isdigit():
            expires_at = int(expires_at_str)
            if now < (expires_at - buffer_seconds):
                should_refresh = False

        if not should_refresh:
            # Nothing to do
            return 0

        updates = refresh_token(client_id=client_id, client_secret=client_secret, scope=scope)

        new_lines = render_env(parsed, kv, updates)
        atomic_write(path, new_lines, mode=0o600)

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
