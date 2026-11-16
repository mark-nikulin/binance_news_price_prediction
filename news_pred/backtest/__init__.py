"""Backtest contract and adapters."""

from news_pred.backtest.backtest_strat_adapter import BacktestStratAdapter, NewsTrendStrategy
from news_pred.backtest.contract import BARS_COLUMNS, BacktestRunResult, BaseBacktester, SIGNAL_SCHEMA

__all__ = [
    "BARS_COLUMNS",
    "BacktestRunResult",
    "BaseBacktester",
    "SIGNAL_SCHEMA",
    "BacktestStratAdapter",
    "NewsTrendStrategy",
]

