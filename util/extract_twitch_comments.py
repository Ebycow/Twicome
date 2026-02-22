#!/usr/bin/env python3
# extract_twitch_comments.py
import argparse
import csv
import json
import sys
from collections import Counter
from typing import Any, Dict, Iterable, Iterator, Optional

def iter_comments(root: Any) -> Iterator[Dict[str, Any]]:
    """comments.json の中のコメント配列を取り出して順に返す。"""
    if isinstance(root, dict) and isinstance(root.get("comments"), list):
        yield from root["comments"]
        return
    if isinstance(root, list):
        # もし JSON がコメント配列そのものだった場合にも対応
        for item in root:
            if isinstance(item, dict):
                yield item
        return
    raise ValueError("Unsupported JSON shape: expected {'comments':[...]} or [...]")

def normalize_name(s: Optional[str], ignore_case: bool) -> Optional[str]:
    if s is None:
        return None
    return s.lower() if ignore_case else s

def to_row(c: Dict[str, Any]) -> Dict[str, Any]:
    commenter = c.get("commenter") or {}
    message = c.get("message") or {}
    return {
        "created_at": c.get("created_at"),
        "content_offset_seconds": c.get("content_offset_seconds"),
        "name": commenter.get("name"),
        "display_name": commenter.get("display_name"),
        "body": message.get("body"),
        "_id": c.get("_id"),
    }

def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract Twitch VOD comments by commenter.name from comments.json"
    )
    p.add_argument("name", nargs="?", help="抽出したい commenter.name（例: jinx_pp）")
    p.add_argument("-i", "--input", default="comments.json", help="入力JSON (default: comments.json)")
    p.add_argument("-o", "--output", default="-", help="出力先 (default: stdout) 例: out.csv")
    p.add_argument("--format", choices=["text", "jsonl", "csv"], default="text",
                   help="出力形式 (default: text)")
    p.add_argument("--ignore-case", action="store_true", help="name の大小文字を無視")
    p.add_argument("--list-names", action="store_true",
                   help="含まれる commenter.name を出現回数つきで表示して終了")
    p.add_argument("--top", type=int, default=50, help="--list-names の表示上位件数 (default: 50)")
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        root = json.load(f)

    comments = list(iter_comments(root))

    if args.list_names:
        cnt = Counter((c.get("commenter") or {}).get("name") for c in comments)
        # None を除外して多い順に表示
        items = [(k, v) for k, v in cnt.items() if k is not None]
        items.sort(key=lambda kv: kv[1], reverse=True)
        for name, n in items[: args.top]:
            print(f"{name}\t{n}")
        return

    if not args.name:
        p.error("name が必要です（または --list-names を使ってください）")

    target = normalize_name(args.name, args.ignore_case)

    filtered = []
    for c in comments:
        n = normalize_name((c.get("commenter") or {}).get("name"), args.ignore_case)
        if n == target:
            filtered.append(to_row(c))

    out_fh = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8", newline="")
    try:
        if args.format == "text":
            for r in filtered:
                off = r.get("content_offset_seconds")
                dn = r.get("display_name") or ""
                nm = r.get("name") or ""
                body = r.get("body") or ""
                print(f"[{off:>6}] {dn} ({nm}): {body}", file=out_fh)
        elif args.format == "jsonl":
            for r in filtered:
                out_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        elif args.format == "csv":
            fieldnames = ["created_at", "content_offset_seconds", "name", "display_name", "body", "_id"]
            w = csv.DictWriter(out_fh, fieldnames=fieldnames)
            w.writeheader()
            for r in filtered:
                w.writerow({k: r.get(k) for k in fieldnames})
    finally:
        if out_fh is not sys.stdout:
            out_fh.close()

if __name__ == "__main__":
    main()
