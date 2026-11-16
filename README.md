# Binance News Price Prediction

Research and trading-signal pipeline for evaluating whether cryptocurrency exchange announcements (starting with **Binance Announcements**) predict short- to medium-term asset returns (LFT/MFT: 5m, 1h, 4h, 24h). Market reactions are measured strictly causally on **OKX** 5-minute bars (`SPOT` and `USDT-SWAP`) aligned by publication timestamp (`published_ts`) with zero look-ahead.

## What Is Implemented

- **News Ingestion & Normalization (`news_pred/providers/`, `news_pred/normalizers/`)**: Fetches Binance announcements idempotently, extracts affected base assets, and classifies events (`listing`, `delisting`, `futures`, `other`) with UTC nanosecond timestamps.
- **OKX Market Data & Crypto Universe Filter (`news_pred/features/market.py`)**: Maps normalized `ASSET-USDT` pairs to active OKX `SPOT` / `SWAP` instruments restricted to `instCategory == "1"` (crypto only, excluding tokenized equities, ETFs, and commodities), and caches merged `±48h` per-event bar windows to Parquet.
- **Causal Event Study (`news_pred/features/event_study.py`)**: Measures immediate impulse (`0 → +5m`) and post-announcement drift (`+60m`, `+240m`, `+1440m`) after an execution delay.
- **Leak-Free Feature & NLP Pipeline (`news_pred/features/dataset.py`, `text.py`)**:
  - Pre-event market features (trailing returns, realized volatility, volume spike).
  - Event metadata (`event_type`).
  - Title embeddings via `sentence-transformers/all-MiniLM-L6-v2` with a leak-free `TextPCATransformer` fit strictly on the training split.
- **Models & Zero-ML Benchmark (`news_pred/models/`)**:
  - `CatBoostTrendModel` (`Huber` loss) evaluated across three feature sets (`market_only`, `market_plus_news`, `market_plus_news_text`) on chronological train/test splits.
  - Zero-ML rule baseline (`SHORT` on `listing` and `futures` after entry delay).
- **Pluggable Backtester Contract (`news_pred/backtest/`, [docs/backtest-contract.md](docs/backtest-contract.md))**: Decoupled `BaseBacktester` / `SIGNAL_SCHEMA` / `BARS_COLUMNS` interface with a reference adapter for `backtest_strat`.

## Experiment Results

All backtests use `\$1,000` notional per trade, `taker_fee = 10 bps`, `half_spread = 2 bps`, `entry_delay = 5 min`, and `horizon = 240 min` unless noted otherwise.

### 1. 50-Symbol MVP Progression (test split from `2026-04-30`)

| Configuration | `train / test` | `test MAE` | `test R²` | `Dir. Accuracy` | Portfolio `net_pnl` | `fees` |
|---|---:|---:|---:|---:|---:|---:|
| `market_only` | 55 / 24 | 0.0459 | -1.04 | 37.5% | -\$255.10 | \$39.73 |
| `market_plus_news` (`Huber` + `event_type`) | 55 / 24 | 0.0341 | -0.25 | 50.0% | -\$45.94 | \$31.85 |
| **`market_plus_news_text` (`MiniLM + PCA`)** | **55 / 24** | **0.0310** | **-0.07** | **70.8%** | **+\$196.81** | **\$35.75** |

Zero-ML benchmark (`SHORT` on `listing` and `futures`, 39 symbols): **+\$1,339.90** (4h) and **+\$2,277.03** (24h).

### 2. Full OKX Crypto Universe (`instCategory == "1"`, 384 matched news rows, test split from `2026-03-05`)

Scaling to all **252** raw OKX matches initially introduced tokenized stocks, ETFs, and commodities (`AAPL`, `NVDA`, `SPY`, `QQQ`, `XAU`, `XAG`, etc.), degrading `market_plus_news_text` to `49.4%` `Dir. Accuracy` (`-\$574.80` net PnL). Filtering strictly by OKX `instCategory == "1"` restored positive signal:

| Configuration (Full Crypto Universe) | `train / test` | `test MAE` | `test R²` | `Dir. Accuracy` | Portfolio `net_pnl` | `fees` | `Sharpe` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `market_only` | 129 / 56 | 0.0537 | -0.004 | 57.1% | +\$31.06 | \$86.27 | 0.06 |
| `market_plus_news` | 129 / 56 | 0.0532 | +0.001 | 58.9% | -\$99.27 | \$67.93 | -0.20 |
| **`market_plus_news_text` (`MiniLM + PCA`)** | **129 / 56** | **0.0536** | **-0.012** | **58.9%** | **+\$161.06** | **\$87.60** | **0.46** |

Zero-ML benchmark on the full crypto universe (**93** symbols): **+\$729.12** (`Sharpe = 0.21`, 4h, 246 fills) and **+\$583.68** (`Sharpe = 0.04`, 24h, 230 fills; **+\$2,137.08** on the top 40 most active symbols).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Optional: plug in your backtester (see requirements-backtest.example.txt & docs/backtest-contract.md)
cp .env.example .env

# 1. Fetch and normalize Binance announcements
np-cli ingest --pages 30 --page-size 20

# 2. Run causal Event Study on OKX bars
np-cli event-study --delay-min 5 --max-symbols 500

# 3. Train, infer, and backtest ML pipeline
np-cli run-mvp --mode market_plus_news_text --horizon-min 240 --notional-usd 1000 --max-symbols 500 --cache-only

# 4. Run zero-ML rule benchmark
np-cli rule-backtest --horizon-min 240 --notional-usd 1000 --max-symbols 500 --cache-only
```

Run tests:

```bash
.venv/bin/pytest -v
```

## Backtester Contract

To use your own backtesting engine instead of `backtest_strat`, follow [`docs/backtest-contract.md`](docs/backtest-contract.md) and subclass `BaseBacktester` in [`news_pred/backtest/contract.py`](news_pred/backtest/contract.py).

## Roadmap & Next Steps

1. **Risk & Liquidity Management**: Add minimum pre-event volume filters, volatility-adjusted position sizing, and protective stop-losses to eliminate illiquid tail-risk outliers.
2. **Strict Out-of-Sample & Walk-Forward Validation**: Add an `--oos-only` backtest mode (`ts >= split_ts`) and rolling walk-forward evaluation across horizons (`60m / 240m / 1440m`).
3. **Dataset & Feature Expansion**: Deeper historical Binance ingestion, additional news providers (e.g., OKX / Bybit announcements), `BTC-USDT` market-regime context features, and `SPOT` vs `SWAP` availability flags.
4. **Server Deployment & `n8n` Orchestration**: Package the pipeline behind a lightweight FastAPI job runner (`POST /runs`, `GET /runs/{run_id}`) on a dedicated server so `n8n` can schedule ingestion, trigger long-running backtests/retraining, and send alerts on completion.
