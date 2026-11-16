"""Binance Announcements provider via public Binance CMS API."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import httpx

from news_pred.paths import RAW_NEWS_DIR
from news_pred.providers.base import BaseNewsProvider, RawNewsItem

BINANCE_CMS_LIST_URL = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
BINANCE_ARTICLE_BASE_URL = "https://www.binance.com/en/support/announcement"

# Default Binance announcement catalogs relevant for market-moving events
DEFAULT_CATALOGS: Dict[int, str] = {
    48: "new_listing",
    161: "delisting",
    49: "latest_news",
}


class BinanceAnnouncementsProvider(BaseNewsProvider):
    """Fetches announcements from Binance CMS API."""

    provider_name = "binance"

    def __init__(
        self,
        raw_dir: Path = RAW_NEWS_DIR,
        catalogs: Optional[Dict[int, str]] = None,
        timeout: float = 15.0,
        max_retries: int = 3,
    ):
        super().__init__(raw_dir=raw_dir)
        self.catalogs = catalogs or DEFAULT_CATALOGS
        self.timeout = timeout
        self.max_retries = max_retries

    def _get_json(self, client: httpx.Client, params: dict) -> dict:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = client.get(BINANCE_CMS_LIST_URL, params=params)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_err = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * attempt)
        raise RuntimeError(f"Binance CMS request failed after {self.max_retries} attempts: {last_err}")

    def fetch(
        self,
        max_pages: int = 5,
        page_size: int = 20,
        catalog_ids: Optional[Sequence[int]] = None,
    ) -> List[RawNewsItem]:
        """Fetch articles across selected Binance announcement catalogs."""
        target_catalogs = list(catalog_ids) if catalog_ids is not None else list(self.catalogs.keys())
        items: List[RawNewsItem] = []

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "lang": "en",
        }
        with httpx.Client(timeout=self.timeout, headers=headers) as client:
            for cat_id in target_catalogs:
                cat_label = self.catalogs.get(cat_id, str(cat_id))
                for page in range(1, max_pages + 1):
                    payload = self._get_json(
                        client,
                        params={"type": 1, "catalogId": cat_id, "pageNo": page, "pageSize": page_size},
                    )
                    catalogs_data = (payload.get("data") or {}).get("catalogs") or []
                    articles = catalogs_data[0].get("articles", []) if catalogs_data else []
                    if not articles:
                        break
                    for art in articles:
                        code = str(art.get("code") or art.get("id"))
                        items.append(
                            RawNewsItem(
                                news_id=code,
                                provider=self.provider_name,
                                published_ms=int(art["releaseDate"]),
                                title=str(art.get("title", "")).strip(),
                                body="",
                                category=cat_label,
                                url=f"{BINANCE_ARTICLE_BASE_URL}/{code}",
                                extra={"catalog_id": cat_id, "article_id": art.get("id")},
                            )
                        )
                    if len(articles) < page_size:
                        break
        return items

