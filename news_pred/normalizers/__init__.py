"""News normalizers."""

from news_pred.normalizers.binance import BinanceNewsNormalizer
from news_pred.normalizers.schema import EventType, NEWS_SCHEMA

__all__ = ["BinanceNewsNormalizer", "EventType", "NEWS_SCHEMA"]

