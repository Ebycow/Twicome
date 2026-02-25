import os

# 例: mysql+pymysql://user:password@dbhost:3306/appdb?charset=utf8mb4
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Set DATABASE_URL in .env or environment variables.")

DEFAULT_PLATFORM = os.getenv("DEFAULT_PLATFORM", "twitch")
ROOT_PATH = os.getenv("ROOT_PATH", "/twicome")
DEFAULT_LOGIN = os.getenv("DEFAULT_LOGIN", "").strip()


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_csv_env(name: str):
    raw = os.getenv(name, "")
    items = []
    seen = set()
    for token in raw.split(","):
        value = token.strip()
        if not value or value in seen:
            continue
        items.append(value)
        seen.add(value)
    return items


# .env 例: QUICK_LINK_LOGINS=userloginid
QUICK_LINK_LOGINS = _parse_csv_env("QUICK_LINK_LOGINS")

# .env 例: HOST_CHECK_ENABLED=true
HOST_CHECK_ENABLED = _parse_bool_env("HOST_CHECK_ENABLED", True)

# .env 例: FAISS_API_URL=http://faiss-api:8100
# 未設定の場合は埋め込み検索機能が無効化される
FAISS_API_URL: str = os.getenv("FAISS_API_URL", "").strip().rstrip("/")
FAISS_ENABLED: bool = bool(FAISS_API_URL)
