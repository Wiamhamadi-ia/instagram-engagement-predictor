"""Collecte les posts des comptes listés dans data/raw/accounts.csv.

Usage:
    python -m src.collection.collect                # tous les comptes
    python -m src.collection.collect --max-posts 100
    python -m src.collection.collect --own          # ton propre compte
Reprise automatique : un compte déjà collecté est ignoré.
"""
import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .client import GraphError, InstagramClient

RAW = Path("data/raw")
OUT = RAW / "posts.jsonl"
PROFILES = RAW / "profiles.jsonl"
FAILED = RAW / "failed_accounts.csv"


def already_done():
    if not PROFILES.exists():
        return set()
    return {json.loads(l)["username"].lower() for l in PROFILES.open(encoding="utf-8")}


def append(path, rows):
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-posts", type=int, default=200)
    ap.add_argument("--own", action="store_true")
    ap.add_argument("--sleep", type=float, default=2.0)
    args = ap.parse_args()

    client = InstagramClient()
    now = datetime.now(timezone.utc).isoformat()

    if args.own:
        posts = client.own_media(args.max_posts)
        append(OUT, [{**p, "account": "OWN", "theme": "melange", "collected_at": now} for p in posts])
        append(PROFILES, [{**client.own_profile(), "theme": "melange", "collected_at": now}])
        print(f"Compte perso : {len(posts)} posts")
        return

    accounts = list(csv.DictReader((RAW / "accounts.csv").open(encoding="utf-8")))
    done = already_done()
    for acc in accounts:
        user = acc["username"]
        if user.lower() in done:
            print(f"[skip] {user}")
            continue
        try:
            profile, posts = client.discover_account(user, args.max_posts)
        except GraphError as e:
            # typique : compte personnel (non Business/Creator) ou privé
            print(f"[fail] {user}: {e}")
            append_failed = not FAILED.exists()
            with FAILED.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if append_failed:
                    w.writerow(["username", "error"])
                w.writerow([user, str(e)])
            continue
        append(OUT, [{**p, "account": user, "theme": acc["theme"], "collected_at": now} for p in posts])
        append(PROFILES, [{**profile, "theme": acc["theme"], "collected_at": now}])
        print(f"[ok]   {user}: {len(posts)} posts")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
