"""Event Study: measure post-announcement price reaction and drift without ML."""

from typing import Dict, Sequence

import numpy as np
import pandas as pd


def run_event_study(
    news_df: pd.DataFrame,
    bars_by_symbol: Dict[str, pd.DataFrame],
    entry_delay_minutes: int = 5,
    horizons_minutes: Sequence[int] = (60, 240, 1440),
    bar_interval_minutes: int = 1,
) -> pd.DataFrame:
    """
    For each news item, align `published_ts` to the latest closed bar (`close_ts <= published_ts`),
    then measure:
      - `imm_ret`: immediate reaction from t0 to t0 + entry_delay
      - `drift_{H}m`: post-announcement drift from (t0 + entry_delay) to (t0 + entry_delay + H)
    """
    records = []
    interval = pd.Timedelta(minutes=bar_interval_minutes)
    delay = pd.Timedelta(minutes=entry_delay_minutes)

    for symbol, grp in news_df.groupby("okx_symbol"):
        bars = bars_by_symbol.get(symbol)
        if bars is None or bars.empty:
            continue

        b = bars.sort_values("bar_ts").copy()
        b["bar_ts"] = pd.to_datetime(b["bar_ts"], utc=True).dt.as_unit("ns")
        b["close_ts"] = b["bar_ts"] + interval
        b = b[["close_ts", "close"]].dropna()

        g = grp.sort_values("published_ts").copy()
        g["published_ts"] = pd.to_datetime(g["published_ts"], utc=True).dt.as_unit("ns")
        # Price known at or before published_ts
        at_pub = pd.merge_asof(
            g[["news_id", "published_ts", "event_type", "okx_symbol", "title"]],
            b.rename(columns={"close_ts": "pub_bar_ts", "close": "px_t0"}),
            left_on="published_ts",
            right_on="pub_bar_ts",
            direction="backward",
            tolerance=pd.Timedelta(minutes=30),
        )
        at_pub["entry_ts"] = at_pub["published_ts"] + delay

        # Entry price at first closed bar >= entry_ts
        at_entry = pd.merge_asof(
            at_pub.sort_values("entry_ts"),
            b.rename(columns={"close_ts": "entry_bar_ts", "close": "px_entry"}),
            left_on="entry_ts",
            right_on="entry_bar_ts",
            direction="forward",
            tolerance=pd.Timedelta(minutes=30),
        )

        for h in horizons_minutes:
            at_entry[f"exit_ts_{h}m"] = at_entry["entry_ts"] + pd.Timedelta(minutes=h)
            merged_h = pd.merge_asof(
                at_entry[[f"exit_ts_{h}m"]].sort_values(f"exit_ts_{h}m"),
                b.rename(columns={"close_ts": f"exit_bar_ts_{h}m", "close": f"px_exit_{h}m"}),
                left_on=f"exit_ts_{h}m",
                right_on=f"exit_bar_ts_{h}m",
                direction="forward",
                tolerance=pd.Timedelta(minutes=30),
            )
            at_entry = at_entry.sort_values(f"exit_ts_{h}m").reset_index(drop=True)
            at_entry[f"px_exit_{h}m"] = merged_h[f"px_exit_{h}m"].values

        at_entry["imm_ret"] = at_entry["px_entry"] / at_entry["px_t0"] - 1.0
        for h in horizons_minutes:
            at_entry[f"drift_{h}m"] = at_entry[f"px_exit_{h}m"] / at_entry["px_entry"] - 1.0

        records.append(at_entry)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True).sort_values("published_ts").reset_index(drop=True)


def summarize_event_study(study_df: pd.DataFrame, horizons_minutes: Sequence[int] = (60, 240, 1440)) -> pd.DataFrame:
    """Aggregate mean/median post-announcement drift and hit-rate by event_type."""
    if study_df.empty:
        return pd.DataFrame()

    rows = []
    for event_type, grp in study_df.groupby("event_type"):
        row = {
            "event_type": event_type,
            "n_events": len(grp),
            "imm_ret_mean_bps": grp["imm_ret"].mean() * 1e4,
        }
        for h in horizons_minutes:
            col = f"drift_{h}m"
            valid = grp[col].dropna()
            row[f"drift_{h}m_n"] = len(valid)
            row[f"drift_{h}m_mean_bps"] = valid.mean() * 1e4 if len(valid) else np.nan
            row[f"drift_{h}m_med_bps"] = valid.median() * 1e4 if len(valid) else np.nan
            row[f"drift_{h}m_pos_rate"] = (valid > 0).mean() if len(valid) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_events", ascending=False).reset_index(drop=True)

