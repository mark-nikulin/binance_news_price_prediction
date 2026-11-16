"""Base interface for news providers (mirrors BaseExchangeFetcher in backtest_strat)."""

import abc
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from news_pred.paths import RAW_NEWS_DIR


@dataclass
class RawNewsItem:
    """Single raw news item returned by a provider."""

    news_id: str
    provider: str
    published_ms: int
    title: str
    body: str = ""
    category: str = ""
    url: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseNewsProvider(abc.ABC):
    """Abstract base class for fetching and persisting raw news."""

    provider_name: str = ""

    def __init__(self, raw_dir: Path = RAW_NEWS_DIR):
        self.raw_dir = Path(raw_dir) / self.provider_name
        self.logger = logging.getLogger(self.__class__.__name__)

    @abc.abstractmethod
    def fetch(self, max_pages: int = 5, page_size: int = 20, **kwargs) -> List[RawNewsItem]:
        """Fetch news items from the upstream source."""

    def save_raw(self, items: List[RawNewsItem], filename: str = "articles.jsonl") -> Path:
        """Persist raw news items idempotently (deduplicated by news_id, atomic replace)."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        target = self.raw_dir / filename

        existing: Dict[str, dict] = {}
        if target.exists():
            with target.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    existing[str(rec["news_id"])] = rec

        for item in items:
            existing[str(item.news_id)] = asdict(item)

        merged = sorted(existing.values(), key=lambda r: r["published_ms"], reverse=True)
        tmp = target.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in merged:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(target)
        self.logger.info("Saved %d raw news items to %s", len(merged), target)
        return target

    def load_raw(self, filename: str = "articles.jsonl") -> List[RawNewsItem]:
        """Load previously saved raw news items."""
        target = self.raw_dir / filename
        if not target.exists():
            return []
        items: List[RawNewsItem] = []
        with target.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(RawNewsItem(**json.loads(line)))
        return items

