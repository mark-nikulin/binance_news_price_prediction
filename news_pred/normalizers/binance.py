"""Normalizer for Binance announcements: event classification and ticker extraction."""

import re
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from news_pred.normalizers.schema import EventType, NEWS_SCHEMA
from news_pred.paths import NORMALIZED_NEWS_DIR
from news_pred.providers.base import RawNewsItem

# Common quote currencies / false-positive uppercase tokens in Binance titles
QUOTE_OR_STOPWORDS: Set[str] = {
    "USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USD", "EUR", "TRY", "BRL",
    "BNB", "API", "VIP", "OTC", "APR", "APY", "UTC", "AMA", "NFT", "ETF", "SEC",
    "USDS", "COIN", "SPOT", "CROSS", "ISOLATED", "NEW", "WILL", "LIST", "DELIST",
}

_PAREN_TICKER_RE = re.compile(r"\(([A-Z0-9]{2,12})\)")
_PAIR_TICKER_RE = re.compile(r"\b([A-Z0-9]{2,12})(?:USDT|USDC|FDUSD|BUSD|BTC|ETH|/USDT|/BTC)\b")
_WORD_TICKER_RE = re.compile(r"\b([A-Z]{2,10})\b")


def classify_binance_event(title: str, category: str = "") -> EventType:
    """Classify a Binance announcement into a market-relevant EventType."""
    low = title.lower()
    cat = category.lower()

    if "delist" in low or "will remove" in low or "removal of" in low or cat == "delisting":
        return EventType.DELISTING
    if any(k in low for k in ("futures", "perpetual", "usds-m", "coin-m", "margin")):
        return EventType.FUTURES
    if any(k in low for k in ("will list", "binance lists", "launchpool", "megadrop", "hodler airdrops")) or cat == "new_listing":
        return EventType.LISTING
    if any(k in low for k in ("network upgrade", "hard fork", "maintenance", "token swap", "rebrand", "redenomination")):
        return EventType.UPGRADE
    return EventType.OTHER


def extract_base_assets(title: str) -> List[str]:
    """Extract unique base asset tickers from a Binance announcement title."""
    found: List[str] = []

    for pat in (_PAREN_TICKER_RE, _PAIR_TICKER_RE):
        for match in pat.findall(title):
            if match not in QUOTE_OR_STOPWORDS and match not in found:
                found.append(match)

    # Fallback: uppercase tokens after "delist" or "list" if nothing in parens/pairs
    if not found:
        for match in _WORD_TICKER_RE.findall(title):
            if match not in QUOTE_OR_STOPWORDS and match not in found:
                found.append(match)

    return found


class BinanceNewsNormalizer:
    """Converts raw Binance announcements into normalized NEWS_SCHEMA Parquet rows."""

    def __init__(self, out_dir: Path = NORMALIZED_NEWS_DIR, quote: str = "USDT"):
        self.out_dir = Path(out_dir)
        self.quote = quote

    def normalize(self, items: Iterable[RawNewsItem]) -> pd.DataFrame:
        rows = []
        for item in items:
            event_type = classify_binance_event(item.title, item.category)
            assets = extract_base_assets(item.title)
            if not assets:
                continue
            ts = pd.to_datetime(item.published_ms, unit="ms", utc=True)
            for asset in assets:
                rows.append(
                    {
                        "news_id": str(item.news_id),
                        "provider": item.provider,
                        "published_ts": ts,
                        "event_type": event_type.value,
                        "base_asset": asset,
                        "okx_symbol": f"{asset}-{self.quote}",
                        "title": item.title,
                        "url": item.url,
                    }
                )
        if not rows:
            return pd.DataFrame(columns=NEWS_SCHEMA.names)
        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["news_id", "okx_symbol"]).sort_values("published_ts").reset_index(drop=True)
        return df

    def save(self, df: pd.DataFrame, filename: str = "binance.parquet") -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        target = self.out_dir / filename
        table = pa.Table.from_pandas(df, schema=NEWS_SCHEMA, preserve_index=False)
        tmp = target.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        tmp.replace(target)
        return target

