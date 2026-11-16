"""Backtester and Signal contracts so any backtester can be plugged into the model pipeline."""

import abc
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa

TS_TYPE = pa.timestamp("ns", tz="UTC")

# Contract 1: Signals produced by model inference and consumed by any backtester
SIGNAL_SCHEMA = pa.schema(
    [
        pa.field("signal_ts", TS_TYPE, nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("direction", pa.int8(), nullable=False),   # +1 long, -1 short, 0 flat
        pa.field("score", pa.float64(), nullable=False),    # predicted return or confidence
        pa.field("horizon_ns", pa.int64(), nullable=False), # holding horizon in nanoseconds
        pa.field("news_id", pa.string(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
    ]
)

# Contract 2: Market bars required by feature builders and bar-based backtesters
BARS_COLUMNS = ["bar_ts", "open", "high", "low", "close", "volume"]


@dataclass
class BacktestRunResult:
    """Standardized backtest summary returned by any BaseBacktester implementation."""

    symbol: str
    signals_count: int
    orders_sent: int
    fills_count: int
    realized_pnl: float
    fees: float
    net_pnl: float
    final_equity: float
    max_drawdown: float
    wall_seconds: float
    engine_name: str = "unknown"
    sharpe_ratio: float = 0.0
    equity_curve: Optional[List[Tuple[int, float]]] = field(default=None, repr=False)


def _daily_pnl_series(equity_curve: Sequence[Tuple[int, float]]) -> pd.Series:
    """Convert `(ts_ns, equity)` points to daily PnL increments in UTC."""
    if not equity_curve:
        return pd.Series(dtype=float)
    ts = pd.to_datetime([pt[0] for pt in equity_curve], unit="ns", utc=True)
    vals = [float(pt[1]) for pt in equity_curve]
    s = pd.Series(vals, index=ts).groupby(level=0).last().sort_index()
    daily_eq = s.resample("1D").last().ffill()
    if len(daily_eq) < 2:
        return pd.Series(dtype=float)
    # Prepend 0.0 before first day so the first day's PnL is captured in diff()
    first_val = float(daily_eq.iloc[0])
    daily_pnl = daily_eq.diff()
    daily_pnl.iloc[0] = first_val
    return daily_pnl.dropna()


def compute_sharpe_from_equity_curve(
    equity_curve: Optional[Sequence[Tuple[int, float]]],
    periods_per_year: float = 365.0,
) -> float:
    """Annualized Sharpe ratio computed from daily PnL increments of an equity curve."""
    if not equity_curve:
        return 0.0
    daily_pnl = _daily_pnl_series(equity_curve)
    if len(daily_pnl) < 2:
        return 0.0
    std = float(daily_pnl.std(ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(daily_pnl.mean() / std * np.sqrt(periods_per_year))


def compute_portfolio_sharpe(
    results: Sequence[BacktestRunResult],
    periods_per_year: float = 365.0,
) -> float:
    """Aggregate symbol equity curves onto a common daily UTC grid and compute portfolio Sharpe."""
    daily_curves = []
    for r in results:
        if not r.equity_curve:
            continue
        ts = pd.to_datetime([pt[0] for pt in r.equity_curve], unit="ns", utc=True)
        vals = [float(pt[1]) for pt in r.equity_curve]
        s = pd.Series(vals, index=ts).groupby(level=0).last().sort_index()
        daily_curves.append(s.resample("1D").last().ffill())
    if not daily_curves:
        return 0.0
    panel = pd.concat(daily_curves, axis=1).sort_index().ffill().fillna(0.0)
    total_eq = panel.sum(axis=1)
    if len(total_eq) < 2:
        return 0.0
    daily_pnl = total_eq.diff()
    daily_pnl.iloc[0] = float(total_eq.iloc[0])
    daily_pnl = daily_pnl.dropna()
    if len(daily_pnl) < 2:
        return 0.0
    std = float(daily_pnl.std(ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(daily_pnl.mean() / std * np.sqrt(periods_per_year))


class BaseBacktester(abc.ABC):
    """
    Abstract backtester contract.

    To plug in a custom backtester, subclass BaseBacktester and implement `run(...)`.
    `signals` must conform to SIGNAL_SCHEMA; `bars` must contain BARS_COLUMNS
    with `bar_ts` as UTC timestamps representing bar start times.
    """

    engine_name: str = "base"

    @abc.abstractmethod
    def run(
        self,
        symbol: str,
        bars: pd.DataFrame,
        signals: pd.DataFrame,
        position_size: float = 1.0,
        notional_usd: Optional[float] = None,
        taker_fee_bps: float = 10.0,
        half_spread_bps: float = 2.0,
        tick_size: Optional[float] = None,
    ) -> BacktestRunResult:
        """Execute signals on historical bars and return standardized BacktestRunResult."""

