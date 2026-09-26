import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

MEDIA_FIELDS = (
    "id,caption,media_type,media_product_type,media_url,thumbnail_url,"
    "permalink,timestamp,like_count,comments_count"
)


class GraphError(Exception):
    pass


class InstagramClient:
    def __init__(self):
        self.token = os.environ["IG_ACCESS_TOKEN"]
        self.user_id = os.environ["IG_USER_ID"]
        version = os.getenv("GRAPH_API_VERSION", "v21.0")
        self.base = f"https://graph.facebook.com/{version}"

    def _get(self, url, params, retries=3):
        params = {**params, "access_token": self.token}
        for attempt in range(retries):
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            err = r.json().get("error", {})
            # 4 / 17 / 32 / 613 = limites de débit : on attend puis on réessaie
            if err.get("code") in (4, 17, 32, 613):
                time.sleep(60 * (attempt + 1))
                continue
            raise GraphError(f"{r.status_code} {err.get('message', r.text)}")
        raise GraphError("Limite de débit dépassée après plusieurs tentatives")

    def discover_account(self, username, max_posts=200, page_size=50):
        """Profil + posts d'un compte public Business/Creator via Business Discovery."""
        posts, after, profile = [], None, None
        while len(posts) < max_posts:
            media = f"media.limit({page_size})"
            media += f".after({after})" if after else ""
            media += f"{{{MEDIA_FIELDS}}}"
            fields = (
                f"business_discovery.username({username})"
                f"{{username,name,biography,followers_count,follows_count,media_count,{media}}}"
            )
            data = self._get(f"{self.base}/{self.user_id}", {"fields": fields})
            bd = data["business_discovery"]
            if profile is None:
                profile = {k: v for k, v in bd.items() if k != "media"}
            batch = bd.get("media", {})
            posts.extend(batch.get("data", []))
            after = batch.get("paging", {}).get("cursors", {}).get("after")
            if not after or not batch.get("data"):
                break
        return profile, posts[:max_posts]

    def own_profile(self):
        fields = "username,name,biography,followers_count,follows_count,media_count"
        return self._get(f"{self.base}/{self.user_id}", {"fields": fields})

    def own_media(self, max_posts=500, page_size=50):
        """Posts de ton propre compte (endpoint /media, plus de champs dispo)."""
        posts, url = [], f"{self.base}/{self.user_id}/media"
        params = {"fields": MEDIA_FIELDS, "limit": page_size}
        while url and len(posts) < max_posts:
            data = self._get(url, params)
            posts.extend(data.get("data", []))
            url = data.get("paging", {}).get("next")
            params = {}
        return posts[:max_posts]
