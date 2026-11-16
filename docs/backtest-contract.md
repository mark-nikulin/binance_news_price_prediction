# Backtester & Signal Contract

This project decouples **news ingestion, feature engineering, and signal generation** from **trade execution and portfolio simulation**. Any custom backtesting engine can be plugged into the pipeline by implementing the contract defined in [`news_pred/backtest/contract.py`](../news_pred/backtest/contract.py).

## 1. Signal Contract (`SIGNAL_SCHEMA`)

Model inference (`news_pred/models/inference.py`) and rule-based generators (`news_pred/models/rules.py`) emit a `pandas.DataFrame` / Parquet table conforming to `SIGNAL_SCHEMA`:

| Column | Arrow / Pandas Type | Description |
|---|---|---|
| `signal_ts` | `timestamp[ns, tz=UTC]` | Causal signal timestamp (`published_ts + entry_delay`). **Never** earlier than news availability. |
| `symbol` | `string` | Normalized instrument pair (e.g. `BTC-USDT`). |
| `direction` | `int8` | Target direction: `+1` (long), `-1` (short), `0` (flat / no-op). |
| `score` | `float64` | Predicted forward return or rule confidence score. |
| `horizon_ns` | `int64` | Intended holding duration in nanoseconds (e.g. `240 * 60 * 1_000_000_000` for 4h). |
| `news_id` | `string` | Unique identifier of the triggering news item. |
| `event_type` | `string` | Normalized event category (`listing`, `delisting`, `futures`, `other`). |

### Causality & Execution Rules

- Signals are strictly causal: a signal at `signal_ts` may only be executed on bars whose timestamp is `>= signal_ts`.
- Position exits occur once `now >= signal_ts + horizon_ns` (or earlier if a newer opposite-direction signal arrives).

## 2. Market Bars Contract (`BARS_COLUMNS`)

Market bars passed to any backtester must be a `pandas.DataFrame` sorted ascending by `bar_ts` containing `BARS_COLUMNS`:

| Column | Type | Description |
|---|---|---|
| `bar_ts` | `timestamp[ns, tz=UTC]` | Bar open timestamp in UTC. |
| `open` | `float64` | Bar open price. |
| `high` | `float64` | Bar high price. |
| `low` | `float64` | Bar low price. |
| `close` | `float64` | Bar close price. |
| `volume` | `float64` | Bar volume. |

## 3. Backtester Interface (`BaseBacktester`)

Subclass `BaseBacktester` in `news_pred/backtest/contract.py` and implement `run(...)`:

```python
from typing import Optional
import pandas as pd
from news_pred.backtest.contract import BacktestRunResult, BaseBacktester

class CustomBacktester(BaseBacktester):
    engine_name = "my_custom_engine"

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
        # 1. Simulate fills on `bars` using `signals` with fee/spread assumptions
        # 2. Return standardized BacktestRunResult
        ...
```

### Standardized Output (`BacktestRunResult`)

Every `BaseBacktester.run(...)` call must return a `BacktestRunResult` dataclass:

- `symbol`: str
- `signals_count`: int
- `orders_sent`: int
- `fills_count`: int
- `realized_pnl`: float
- `fees`: float
- `net_pnl`: float (`realized_pnl - fees`)
- `final_equity`: float
- `max_drawdown`: float
- `wall_seconds`: float
- `engine_name`: str
- `sharpe_ratio`: float (optional, annualized daily PnL Sharpe; defaults to `0.0`)
- `equity_curve`: optional `list[tuple[int, float]]` of `(ts_ns, equity)` points used for portfolio-level Sharpe aggregation (`compute_portfolio_sharpe`)

## 4. Reference Adapter

See [`news_pred/backtest/backtest_strat_adapter.py`](../news_pred/backtest/backtest_strat_adapter.py) (`BacktestStratAdapter` + `NewsTrendStrategy`) for a complete working implementation built on top of `backtest_strat`.

To inject a custom backtester into the pipeline (`news_pred/pipeline.py`), pass an instance of your `BaseBacktester` subclass to `run_mvp_pipeline(..., backtester=CustomBacktester())` or `run_rule_backtest_pipeline(..., backtester=CustomBacktester())`.
