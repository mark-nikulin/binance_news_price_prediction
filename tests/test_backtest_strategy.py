"""Integration test for BacktestStratAdapter + NewsTrendStrategy."""

import numpy as np
import pandas as pd

from news_pred.backtest.backtest_strat_adapter import BacktestStratAdapter


def test_backtest_strat_adapter_long_trend():
    # Uptrend from 100 to 120 over 60 minutes
    idx = pd.date_range("2024-01-01 00:00:00", periods=60, freq="1min", tz="UTC")
    close = np.linspace(100.0, 120.0, 60)
    bars = pd.DataFrame(
        {
            "bar_ts": idx,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(60, 100.0),
        }
    )

    signals = pd.DataFrame(
        {
            "signal_ts": [pd.Timestamp("2024-01-01 00:05:00", tz="UTC")],
            "symbol": ["TEST-USDT"],
            "direction": [1],
            "score": [0.05],
            "horizon_ns": [30 * 60 * 1_000_000_000],  # 30 minutes
            "news_id": ["n1"],
            "event_type": ["listing"],
        }
    )

    adapter = BacktestStratAdapter()
    res = adapter.run(
        symbol="TEST-USDT",
        bars=bars,
        signals=signals,
        position_size=1.0,
        taker_fee_bps=10.0,
        half_spread_bps=1.0,
    )
    assert res.engine_name == "backtest_strat"
    assert res.fills_count == 2  # entry + horizon exit
    assert res.net_pnl > 0

