"""Unit tests for Binance news normalizer."""

from news_pred.normalizers.binance import BinanceNewsNormalizer, classify_binance_event, extract_base_assets
from news_pred.normalizers.schema import EventType
from news_pred.providers.base import RawNewsItem


def test_classify_events():
    assert classify_binance_event("Binance Will List Pepe (PEPE) in Innovation Zone") == EventType.LISTING
    assert classify_binance_event("Binance Will Delist OMG, WAVES, XEM on 2024-06-17") == EventType.DELISTING
    assert classify_binance_event("Binance Futures Will Launch USD-M XYZ Perpetual Contract") == EventType.FUTURES
    assert classify_binance_event("Binance Will Support the Arbitrum (ARB) Network Upgrade") == EventType.UPGRADE
    assert classify_binance_event("Binance VIP Program Update for May") == EventType.OTHER


def test_extract_base_assets_and_normalize():
    items = [
        RawNewsItem(
            news_id="a1",
            provider="binance",
            published_ms=1_700_000_000_000,
            title="Binance Will List Celestia (TIA) and Memecoin (MEME)",
            category="new_listing",
        ),
        RawNewsItem(
            news_id="a2",
            provider="binance",
            published_ms=1_700_000_100_000,
            title="Binance Will Delist WAVESUSDT Perpetual Contract",
            category="delisting",
        ),
    ]
    df = BinanceNewsNormalizer().normalize(items)
    assert len(df) == 3
    assert set(df["okx_symbol"]) == {"TIA-USDT", "MEME-USDT", "WAVES-USDT"}
    assert df.loc[df["okx_symbol"] == "WAVES-USDT", "event_type"].iloc[0] == "delisting"

