"""News data providers."""

from news_pred.providers.base import BaseNewsProvider, RawNewsItem
from news_pred.providers.binance import BinanceAnnouncementsProvider

PROVIDERS = {
    "binance": BinanceAnnouncementsProvider,
}


def get_provider(name: str, **kwargs) -> BaseNewsProvider:
    key = name.lower()
    if key not in PROVIDERS:
        raise ValueError(f"Unknown news provider '{name}'. Available: {sorted(PROVIDERS)}")
    return PROVIDERS[key](**kwargs)


__all__ = ["BaseNewsProvider", "RawNewsItem", "BinanceAnnouncementsProvider", "PROVIDERS", "get_provider"]

