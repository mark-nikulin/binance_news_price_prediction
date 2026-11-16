"""Market bar loading (via backtest_strat or OKX REST klines fallback) and causal feature engineering."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Set

import httpx
import numpy as np
import pandas as pd

from news_pred.paths import MARKET_DIR

logger = logging.getLogger(__name__)

OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"
OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments"


def fetch_okx_symbol_map(
    cache_dir: Path = MARKET_DIR,
    crypto_only: bool = True,
    force_refresh: bool = False,
) -> dict[str, str]:
    """
    Build a mapping from normalized `ASSET-USDT` to an active OKX instId.
    Prefers SPOT (`ASSET-USDT`), falling back to USDT-margined SWAP (`ASSET-USDT-SWAP`).
    When `crypto_only=True`, restricts to OKX `instCategory == "1"` (crypto) and excludes
    tokenized equities/ETFs (`"3"`) and commodities (`"4"`).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = "okx_crypto_symbol_map.json" if crypto_only else "okx_symbol_map.json"
    cache_file = cache_dir / cache_name
    if cache_file.exists() and not force_refresh:
        return json.loads(cache_file.read_text(encoding="utf-8"))

    mapping: dict[str, str] = {}
    with httpx.Client(timeout=15.0) as client:
        for inst_type in ("SWAP", "SPOT"):  # SPOT second so it overwrites SWAP when both exist
            resp = client.get(OKX_INSTRUMENTS_URL, params={"instType": inst_type})
            resp.raise_for_status()
            for row in resp.json().get("data") or []:
                if crypto_only and str(row.get("instCategory", "1")) != "1":
                    continue
                inst_id = row.get("instId", "")
                if inst_type == "SPOT" and inst_id.endswith("-USDT"):
                    mapping[inst_id] = inst_id
                elif inst_type == "SWAP" and inst_id.endswith("-USDT-SWAP"):
                    base_pair = inst_id[:-5]  # strip '-SWAP' -> 'ASSET-USDT'
                    mapping[base_pair] = inst_id

    cache_file.write_text(json.dumps(mapping, sort_keys=True), encoding="utf-8")
    return mapping


def fetch_okx_spot_symbols(cache_dir: Path = MARKET_DIR) -> Set[str]:
    """Fetch and cache the set of active SPOT instIds on OKX (e.g. {'BTC-USDT', ...})."""
    return {k for k, v in fetch_okx_symbol_map(cache_dir).items() if v == k}


def fetch_okx_candles(
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    bar: str = "1m",
    cache_dir: Path = MARKET_DIR,
    force_refresh: bool = False,
    cache_only: bool = False,
) -> pd.DataFrame:
    """
    Lightweight OKX REST history-candles fetcher for LFT/MFT research.
    Caches results to Parquet so repeated runs are idempotent.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol}_{bar}.parquet"
    meta_file = cache_dir / f"{symbol}_{bar}.meta.json"

    if cache_file.exists() and not force_refresh:
        cached = pd.read_parquet(cache_file)
        cached["bar_ts"] = pd.to_datetime(cached["bar_ts"], utc=True).dt.as_unit("ns")
        if not cached.empty:
            mask = (cached["bar_ts"] >= start_ts) & (cached["bar_ts"] <= end_ts)
            sub_cached = cached.loc[mask].sort_values("bar_ts").reset_index(drop=True)
            if cache_only or not sub_cached.empty:
                return sub_cached
    if cache_only:
        return pd.DataFrame(columns=["bar_ts", "open", "high", "low", "close", "volume"])

    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
    rows = []
    after_ms = end_ms + 60_000

    with httpx.Client(timeout=15.0) as client:
        while True:
            resp = client.get(
                OKX_CANDLES_URL,
                params={"instId": symbol, "bar": bar, "after": str(after_ms), "limit": "100"},
            )
            resp.raise_for_status()
            data = resp.json().get("data") or []
            if not data:
                break
            for row in data:
                ts_ms = int(row[0])
                if ts_ms < start_ms:
                    continue
                rows.append(
                    {
                        "bar_ts": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                        "volume": float(row[5]),
                    }
                )
            oldest_ms = int(data[-1][0])
            if oldest_ms <= start_ms or oldest_ms >= after_ms:
                break
            after_ms = oldest_ms

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["bar_ts", "open", "high", "low", "close", "volume"])

    df["bar_ts"] = pd.to_datetime(df["bar_ts"], utc=True).dt.as_unit("ns")
    df = df.drop_duplicates(subset=["bar_ts"]).sort_values("bar_ts").reset_index(drop=True)

    if cache_file.exists():
        prev = pd.read_parquet(cache_file)
        prev["bar_ts"] = pd.to_datetime(prev["bar_ts"], utc=True).dt.as_unit("ns")
        df = pd.concat([prev, df], ignore_index=True).drop_duplicates(subset=["bar_ts"]).sort_values("bar_ts").reset_index(drop=True)

    tmp = cache_file.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(cache_file)
    meta_file.write_text(json.dumps({"requested_start_ms": start_ms, "requested_end_ms": end_ms}), encoding="utf-8")

    mask = (df["bar_ts"] >= start_ts) & (df["bar_ts"] <= end_ts)
    return df.loc[mask].reset_index(drop=True)


def load_bars(
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    bar: str = "1m",
    cache_dir: Path = MARKET_DIR,
    backtest_strat_processed_dir: Optional[Path] = None,
    cache_only: bool = False,
) -> pd.DataFrame:
    """
    Load 1m OHLCV bars for `symbol`.
    Tries backtest_strat derived trade_bars first if `backtest_strat_processed_dir` is provided,
    otherwise falls back to cached/REST OKX candles.
    """
    if backtest_strat_processed_dir is not None:
        try:
            from backtest_strat.storage.reader import iter_derived  # type: ignore

            frames = [batch.to_pandas() for batch in iter_derived(str(backtest_strat_processed_dir), "okx", symbol, "trade_bars_1m")]
            if frames:
                df = pd.concat(frames, ignore_index=True)
                ts_col = "bar_ts" if "bar_ts" in df.columns else "ts"
                df["bar_ts"] = pd.to_datetime(df[ts_col], utc=True)
                mask = (df["bar_ts"] >= start_ts) & (df["bar_ts"] <= end_ts)
                cols = [c for c in ["bar_ts", "open", "high", "low", "close", "volume"] if c in df.columns]
                if not df.loc[mask].empty:
                    return df.loc[mask, cols].sort_values("bar_ts").reset_index(drop=True)
        except Exception as exc:
            logger.debug("backtest_strat derived bars unavailable for %s: %s", symbol, exc)

    return fetch_okx_candles(symbol, start_ts=start_ts, end_ts=end_ts, bar=bar, cache_dir=cache_dir, cache_only=cache_only)


def compute_market_features(bars: pd.DataFrame, interval_minutes: int = 1) -> pd.DataFrame:
    """
    Compute causal market features at bar close (`close_ts = bar_ts + interval`).
    All rolling windows use only past and current closed bars (no look-ahead).
    """
    df = bars.sort_values("bar_ts").reset_index(drop=True).copy()
    interval = pd.Timedelta(minutes=interval_minutes)
    df["close_ts"] = df["bar_ts"] + interval

    close = df["close"].astype(float)
    log_ret = np.log(close / close.shift(1))

    def bars_for(minutes: int) -> int:
        return max(1, minutes // interval_minutes)

    df["ret_5m"] = close.pct_change(bars_for(5))
    df["ret_1h"] = close.pct_change(bars_for(60))
    df["ret_4h"] = close.pct_change(bars_for(240))
    df["ret_24h"] = close.pct_change(bars_for(1440))

    df["vol_1h"] = log_ret.rolling(bars_for(60), min_periods=10).std()
    df["vol_24h"] = log_ret.rolling(bars_for(1440), min_periods=60).std()

    vol_1h_sum = df["volume"].rolling(bars_for(60), min_periods=1).sum()
    vol_24h_avg = df["volume"].rolling(bars_for(1440), min_periods=60).mean() * bars_for(60)
    df["rel_volume_1h"] = vol_1h_sum / vol_24h_avg.replace(0, np.nan)

    return df

