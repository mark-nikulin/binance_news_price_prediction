"""Unit tests for causal Event Study alignment."""

import numpy as np
import pandas as pd

from news_pred.features.event_study import run_event_study, summarize_event_study


def test_event_study_post_announcement_drift():
    # 120 one-minute bars starting 2024-01-01 00:00 UTC
    idx = pd.date_range("2024-01-01 00:00:00", periods=120, freq="1min", tz="UTC")
    # Price jumps at t=10..14 (immediate reaction), then drifts up from t=15..75
    close = np.full(120, 100.0)
    close[10:15] = 102.0
    close[15:] = np.linspace(102.0, 110.0, 105)

    bars = pd.DataFrame(
        {
            "bar_ts": idx,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(120, 1000.0),
        }
    )

    news_df = pd.DataFrame(
        {
            "news_id": ["n1"],
            "published_ts": [pd.Timestamp("2024-01-01 00:10:00", tz="UTC")],
            "event_type": ["listing"],
            "okx_symbol": ["TEST-USDT"],
            "title": ["Binance Will List TEST"],
        }
    )

    study = run_event_study(
        news_df,
        {"TEST-USDT": bars},
        entry_delay_minutes=5,
        horizons_minutes=(60,),
    )
    assert len(study) == 1
    assert study["imm_ret"].iloc[0] > 0
    assert study["drift_60m"].iloc[0] > 0

    summary = summarize_event_study(study, horizons_minutes=(60,))
    assert summary.loc[0, "event_type"] == "listing"
    assert summary.loc[0, "drift_60m_mean_bps"] > 0

