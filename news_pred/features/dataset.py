"""Causal dataset builder: asof-join news with market features and forward trend targets."""

from typing import Dict

import numpy as np
import pandas as pd

from news_pred.features.market import compute_market_features

MARKET_FEATURE_COLS = [
    "ret_5m",
    "ret_1h",
    "ret_4h",
    "ret_24h",
    "vol_1h",
    "vol_24h",
    "rel_volume_1h",
]


def build_training_dataset(
    news_df: pd.DataFrame,
    bars_by_symbol: Dict[str, pd.DataFrame],
    entry_delay_minutes: int = 5,
    horizon_minutes: int = 240,
    cost_bps: float = 20.0,
    bar_interval_minutes: int = 1,
) -> pd.DataFrame:
    """
    Build ML-ready dataset with zero look-ahead:
      - features come from bars closed at or before `published_ts`
      - entry happens at `signal_ts = published_ts + entry_delay_minutes`
      - target is forward return from `entry_ts` to `entry_ts + horizon_minutes`
      - `target_dir`: +1 if fwd_ret > cost, -1 if fwd_ret < -cost, 0 otherwise
    """
    delay = pd.Timedelta(minutes=entry_delay_minutes)
    horizon = pd.Timedelta(minutes=horizon_minutes)
    cost_thr = cost_bps / 1e4

    frames = []
    for symbol, grp in news_df.groupby("okx_symbol"):
        bars = bars_by_symbol.get(symbol)
        if bars is None or bars.empty:
            continue

        feat_bars = compute_market_features(bars, interval_minutes=bar_interval_minutes)
        feat_bars["close_ts"] = pd.to_datetime(feat_bars["close_ts"], utc=True).dt.as_unit("ns")
        feat_bars = feat_bars.sort_values("close_ts")

        g = grp.sort_values("published_ts").copy()
        g["published_ts"] = pd.to_datetime(g["published_ts"], utc=True).dt.as_unit("ns")
        joined = pd.merge_asof(
            g,
            feat_bars[["close_ts", "close"] + MARKET_FEATURE_COLS],
            left_on="published_ts",
            right_on="close_ts",
            direction="backward",
            tolerance=pd.Timedelta(minutes=30),
        ).rename(columns={"close": "px_t0"})

        joined["signal_ts"] = joined["published_ts"] + delay
        joined["exit_ts"] = joined["signal_ts"] + horizon

        entry_px = pd.merge_asof(
            joined[["signal_ts"]].sort_values("signal_ts"),
            feat_bars[["close_ts", "close"]].rename(columns={"close_ts": "entry_bar_ts", "close": "px_entry"}),
            left_on="signal_ts",
            right_on="entry_bar_ts",
            direction="forward",
            tolerance=pd.Timedelta(minutes=30),
        )
        joined = joined.sort_values("signal_ts").reset_index(drop=True)
        joined["px_entry"] = entry_px["px_entry"].values

        exit_px = pd.merge_asof(
            joined[["exit_ts"]].sort_values("exit_ts"),
            feat_bars[["close_ts", "close"]].rename(columns={"close_ts": "exit_bar_ts", "close": "px_exit"}),
            left_on="exit_ts",
            right_on="exit_bar_ts",
            direction="forward",
            tolerance=pd.Timedelta(minutes=30),
        )
        joined = joined.sort_values("exit_ts").reset_index(drop=True)
        joined["px_exit"] = exit_px["px_exit"].values

        joined = joined.dropna(subset=["px_t0", "px_entry", "px_exit"])

        joined["fwd_ret"] = joined["px_exit"] / joined["px_entry"] - 1.0
        joined["target_dir"] = np.where(
            joined["fwd_ret"] > cost_thr,
            1,
            np.where(joined["fwd_ret"] < -cost_thr, -1, 0),
        )
        frames.append(joined)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("published_ts").reset_index(drop=True)

