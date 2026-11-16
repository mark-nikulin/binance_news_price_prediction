"""Unified normalized news schema aligned with backtest_strat timestamps."""

import enum
import pyarrow as pa

TS_TYPE = pa.timestamp("ns", tz="UTC")


class EventType(str, enum.Enum):
    LISTING = "listing"
    DELISTING = "delisting"
    FUTURES = "futures"
    UPGRADE = "upgrade"
    OTHER = "other"


NEWS_SCHEMA = pa.schema(
    [
        pa.field("news_id", pa.string(), nullable=False),
        pa.field("provider", pa.string(), nullable=False),
        pa.field("published_ts", TS_TYPE, nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("base_asset", pa.string(), nullable=False),
        pa.field("okx_symbol", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
    ]
)

