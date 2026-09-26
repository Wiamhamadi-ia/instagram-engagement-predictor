"""Télécharge une image par post dans data/raw/media/<compte>/<id>.jpg.

Les URLs Instagram expirent : à lancer juste après la collecte.
Reprise automatique (fichiers existants ignorés), résultats dans media_index.csv.
"""
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

RAW = Path("data/raw")
MEDIA = RAW / "media"
INDEX = RAW / "media_index.csv"


def image_url(post):
    if post["media_type"] == "VIDEO":
        return post.get("thumbnail_url")
    return post.get("media_url")


def fetch(post):
    url = image_url(post)
    path = MEDIA / post["account"] / f"{post['id']}.jpg"
    if path.exists():
        return post["id"], post["account"], "ok"
    if not url:
        return post["id"], post["account"], "no_url"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        return post["id"], post["account"], f"error: {e.__class__.__name__}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    return post["id"], post["account"], "ok"


def main():
    posts = {}
    for line in (RAW / "posts.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        posts[p["id"]] = p
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(fetch, posts.values()))
    with INDEX.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "account", "status"])
        w.writerows(results)
    ok = sum(r[2] == "ok" for r in results)
    print(f"{ok}/{len(results)} images téléchargées")


if __name__ == "__main__":
    main()
