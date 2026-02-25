from __future__ import annotations

import os
import sys

from sqlalchemy import create_engine, inspect

REQUIRED_TABLES = {
    "users",
    "vods",
    "comments",
    "community_notes",
    "vod_ingest_markers",
}


def main() -> int:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        print("DATABASE_URL is not set.", file=sys.stderr)
        return 1

    engine = create_engine(database_url, future=True)
    with engine.connect() as conn:
        tables = set(inspect(conn).get_table_names())

    # テーブル一致の確認しかしていない
    missing = sorted(REQUIRED_TABLES - tables)
    if missing:
        print(f"Missing tables: {', '.join(missing)}", file=sys.stderr)
        return 1

    print("Schema check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
