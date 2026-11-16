"""Rule-based baseline signal generator (no ML) for benchmarking against CatBoost."""

import numpy as np
import pandas as pd

from news_pred.backtest.contract import SIGNAL_SCHEMA


def generate_rule_signals(
    news_df: pd.DataFrame,
    short_events: tuple[str, ...] = ("listing", "futures"),
    long_events: tuple[str, ...] = (),
    entry_delay_minutes: int = 5,
    horizon_minutes: int = 240,
) -> pd.DataFrame:
    """
    Generate deterministic SIGNAL_SCHEMA signals from event_type rules.
    Useful as a zero-ML benchmark to test whether raw Event Study drift survives fees/spread.
    """
    if news_df.empty:
        return pd.DataFrame(columns=SIGNAL_SCHEMA.names)

    ev = news_df["event_type"].astype(str)
    directions = np.where(
        ev.isin(short_events),
        -1,
        np.where(ev.isin(long_events), 1, 0),
    ).astype("int8")

    signal_ts = (
        pd.to_datetime(news_df["published_ts"], utc=True)
        + pd.Timedelta(minutes=entry_delay_minutes)
    )
    horizon_ns = int(horizon_minutes) * 60 * 1_000_000_000

    out = pd.DataFrame(
        {
            "signal_ts": signal_ts,
            "symbol": news_df["okx_symbol"].astype(str),
            "direction": directions,
            "score": directions.astype("float64"),
            "horizon_ns": np.full(len(news_df), horizon_ns, dtype="int64"),
            "news_id": news_df["news_id"].astype(str),
            "event_type": ev,
        }
    )
    return out[out["direction"] != 0].sort_values("signal_ts").reset_index(drop=True)

