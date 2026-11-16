"""Adapter implementing BaseBacktester on top of backtest_strat.BacktestEngine."""

from typing import Optional

import numpy as np
import pandas as pd

from backtest_strat.backtest.bar_feed import bars_to_events
from backtest_strat.backtest.engine import BacktestEngine
from backtest_strat.backtest.strategy import Context, Strategy
from backtest_strat.backtest.types import BUY, SELL, InstrumentSpec

from news_pred.backtest.contract import (
    BacktestRunResult,
    BaseBacktester,
    compute_sharpe_from_equity_curve,
)


class NewsTrendStrategy(Strategy):
    """
    Executes precomputed news-trend signals on bar-derived events.

    Assumes signals are sorted by `signal_ts_ns` and each signal has:
    (signal_ts_ns, direction, horizon_ns).
    """

    def __init__(
        self,
        signals_df: pd.DataFrame,
        position_size: float = 1.0,
        notional_usd: Optional[float] = None,
    ):
        self.position_size = float(position_size)
        self.notional_usd = float(notional_usd) if notional_usd is not None else None
        if signals_df.empty:
            self._signals = []
        else:
            s = signals_df.sort_values("signal_ts").reset_index(drop=True)
            ts_ns = pd.to_datetime(s["signal_ts"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
            dirs = s["direction"].astype(int).to_numpy()
            horizons = s["horizon_ns"].astype("int64").to_numpy()
            self._signals = list(zip(ts_ns, dirs, horizons))
        self._cursor = 0
        self._exit_ts_ns: Optional[int] = None

    def _close_position(self, ctx: Context) -> None:
        pos = ctx.position
        if abs(pos) < ctx.spec.lot_size:
            return
        side = SELL if pos > 0 else BUY
        ctx.market(side, abs(pos), tag="horizon_exit")
        self._exit_ts_ns = None

    def on_book(self, ctx: Context) -> None:
        now = ctx.now

        # 1. Close expired position first
        if self._exit_ts_ns is not None and now >= self._exit_ts_ns:
            self._close_position(ctx)

        # 2. Consume any signals that became active at or before `now`
        latest_sig = None
        while self._cursor < len(self._signals) and self._signals[self._cursor][0] <= now:
            latest_sig = self._signals[self._cursor]
            self._cursor += 1

        if latest_sig is None:
            return

        sig_ts_ns, direction, horizon_ns = latest_sig
        if direction == 0:
            return

        pos = ctx.position
        target_side = BUY if direction > 0 else SELL

        # If already holding opposite direction, close first
        if (pos > 0 and direction < 0) or (pos < 0 and direction > 0):
            self._close_position(ctx)
            pos = 0.0

        if abs(pos) < 1e-9:
            qty = self.position_size
            if self.notional_usd is not None:
                mid = ctx.book.mid()
                if mid is not None and mid > 0:
                    qty = self.notional_usd / mid
            ctx.market(target_side, qty, tag="news_entry")
            self._exit_ts_ns = sig_ts_ns + int(horizon_ns)
        else:
            # Extend horizon on reinforcing signal
            self._exit_ts_ns = max(self._exit_ts_ns or 0, sig_ts_ns + int(horizon_ns))

    def on_end(self, ctx: Context) -> None:
        self._close_position(ctx)


def _compute_max_drawdown(equity_curve: list) -> float:
    if not equity_curve:
        return 0.0
    vals = np.array([eq for _, eq in equity_curve], dtype=float)
    peaks = np.maximum.accumulate(vals)
    drawdowns = vals - peaks
    return float(drawdowns.min()) if len(drawdowns) else 0.0


class BacktestStratAdapter(BaseBacktester):
    """Runs NewsTrendStrategy on bar-derived synthetic L2 events via backtest_strat."""

    engine_name = "backtest_strat"

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
        bar_interval_minutes: int = 1,
    ) -> BacktestRunResult:
        if bars.empty:
            raise ValueError(f"Empty bars DataFrame for {symbol}")

        b = bars.sort_values("bar_ts").reset_index(drop=True)
        ts_ns = pd.to_datetime(b["bar_ts"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
        close = b["close"].astype(float).to_numpy()

        if tick_size is None:
            med_px = float(np.nanmedian(close))
            tick_size = max(round(med_px * 1e-5, 10), 1e-8)

        interval_ns = bar_interval_minutes * 60 * 1_000_000_000
        batch = bars_to_events(
            bar_ts_ns=ts_ns,
            close=close,
            interval_ns=interval_ns,
            tick_size=tick_size,
            half_spread_bps=half_spread_bps,
            fill_gaps=False,
        )

        spec = InstrumentSpec(
            symbol=symbol,
            tick_size=tick_size,
            lot_size=1e-6,
            taker_fee=taker_fee_bps / 1e4,
        )
        sym_signals = signals[signals["symbol"] == symbol] if not signals.empty else signals
        strategy = NewsTrendStrategy(sym_signals, position_size=position_size, notional_usd=notional_usd)
        engine = BacktestEngine(strategy=strategy, spec=spec)
        res = engine.run_batches([batch])

        final_eq = res.portfolio.equity(res.final_mid) if res.final_mid is not None else (
            res.portfolio.realized_pnl - res.portfolio.fees
        )
        net_pnl = res.portfolio.realized_pnl - res.portfolio.fees
        max_dd = _compute_max_drawdown(res.portfolio.equity_curve)
        sharpe = compute_sharpe_from_equity_curve(res.portfolio.equity_curve)

        return BacktestRunResult(
            symbol=symbol,
            signals_count=len(sym_signals),
            orders_sent=res.orders_sent,
            fills_count=len(res.portfolio.fills),
            realized_pnl=float(res.portfolio.realized_pnl),
            fees=float(res.portfolio.fees),
            net_pnl=float(net_pnl),
            final_equity=float(final_eq),
            max_drawdown=max_dd,
            wall_seconds=float(res.wall_seconds),
            engine_name=self.engine_name,
            sharpe_ratio=sharpe,
            equity_curve=list(res.portfolio.equity_curve),
        )
