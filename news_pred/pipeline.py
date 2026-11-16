"""End-to-end pipeline orchestration: ingest -> normalize -> event study -> train -> infer -> backtest."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from news_pred.backtest.backtest_strat_adapter import BacktestStratAdapter
from news_pred.backtest.contract import BacktestRunResult, BaseBacktester
from news_pred.features.dataset import build_training_dataset
from news_pred.features.event_study import run_event_study, summarize_event_study
from news_pred.features.market import fetch_okx_symbol_map, load_bars
from news_pred.models.baseline import CatBoostTrendModel, TrainMetrics
from news_pred.models.inference import generate_signals, save_signals
from news_pred.models.rules import generate_rule_signals
from news_pred.normalizers.binance import BinanceNewsNormalizer
from news_pred.paths import NORMALIZED_NEWS_DIR, REPORTS_DIR
from news_pred.providers import get_provider

logger = logging.getLogger(__name__)


@dataclass
class MVPReport:
    news_count: int
    symbols_count: int
    metrics: TrainMetrics
    backtest_results: List[BacktestRunResult]


def ingest_and_normalize(
    provider_name: str = "binance",
    max_pages: int = 5,
    page_size: int = 20,
) -> pd.DataFrame:
    """Fetch raw news, persist idempotently, normalize, and save Parquet."""
    provider = get_provider(provider_name)
    items = provider.fetch(max_pages=max_pages, page_size=page_size)
    provider.save_raw(items)
    all_items = provider.load_raw()

    normalizer = BinanceNewsNormalizer()
    df = normalizer.normalize(all_items)
    out_path = normalizer.save(df, filename=f"{provider_name}.parquet")
    logger.info("Normalized %d news-asset rows -> %s", len(df), out_path)
    return df


def _merge_event_windows(
    timestamps: pd.Series,
    lookback_hours: int,
    forward_hours: int,
) -> List[tuple[pd.Timestamp, pd.Timestamp]]:
    """Merge overlapping [ts - lookback, ts + forward] windows to avoid downloading multi-year gaps."""
    lb = pd.Timedelta(hours=lookback_hours)
    fw = pd.Timedelta(hours=forward_hours)
    intervals = sorted((pd.Timestamp(t) - lb, pd.Timestamp(t) + fw) for t in timestamps)
    merged: List[tuple[pd.Timestamp, pd.Timestamp]] = []
    for s, e in intervals:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged


def load_bars_for_news(
    news_df: pd.DataFrame,
    lookback_hours: int = 48,
    forward_hours: int = 48,
    bar: str = "5m",
    backtest_strat_processed_dir: Optional[Path] = None,
    max_symbols: Optional[int] = None,
    cache_only: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Load bars for OKX-listed SPOT/SWAP symbols using merged ±48h per-event windows."""
    sym_map = fetch_okx_symbol_map()
    matched = news_df[news_df["okx_symbol"].isin(sym_map)]
    logger.info("Matched %d/%d news rows to active OKX SPOT/SWAP symbols", len(matched), len(news_df))

    bars_by_symbol: Dict[str, pd.DataFrame] = {}
    symbols = list(matched["okx_symbol"].value_counts().index)
    if max_symbols is not None:
        symbols = symbols[:max_symbols]

    for sym in symbols:
        inst_id = sym_map[sym]
        sub = matched[matched["okx_symbol"] == sym]
        windows = _merge_event_windows(sub["published_ts"], lookback_hours, forward_hours)
        frames = []
        for start_ts, end_ts in windows:
            try:
                bars = load_bars(
                    symbol=inst_id,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    bar=bar,
                    backtest_strat_processed_dir=backtest_strat_processed_dir,
                    cache_only=cache_only,
                )
                if not bars.empty:
                    frames.append(bars)
            except Exception as exc:
                logger.warning("Window failed for %s (%s): %s", sym, inst_id, exc)
        if frames:
            combined = (
                pd.concat(frames, ignore_index=True)
                .drop_duplicates(subset=["bar_ts"])
                .sort_values("bar_ts")
                .reset_index(drop=True)
            )
            bars_by_symbol[sym] = combined
    return bars_by_symbol


def run_event_study_pipeline(
    provider_name: str = "binance",
    entry_delay_minutes: int = 5,
    horizons_minutes: Sequence[int] = (60, 240, 1440),
    max_symbols: Optional[int] = 15,
    backtest_strat_processed_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Run Event Study on normalized news and save summary CSV."""
    news_path = NORMALIZED_NEWS_DIR / f"{provider_name}.parquet"
    news_df = pd.read_parquet(news_path)
    news_df["published_ts"] = pd.to_datetime(news_df["published_ts"], utc=True)

    bars_by_symbol = load_bars_for_news(
        news_df,
        backtest_strat_processed_dir=backtest_strat_processed_dir,
        max_symbols=max_symbols,
    )
    study_df = run_event_study(
        news_df,
        bars_by_symbol,
        entry_delay_minutes=entry_delay_minutes,
        horizons_minutes=horizons_minutes,
        bar_interval_minutes=5,
    )
    summary = summarize_event_study(study_df, horizons_minutes=horizons_minutes)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not summary.empty:
        summary.to_csv(REPORTS_DIR / "event_study_summary.csv", index=False)
    return summary


def run_mvp_pipeline(
    provider_name: str = "binance",
    feature_mode: str = "market_plus_news",
    entry_delay_minutes: int = 5,
    horizon_minutes: int = 240,
    score_threshold: float = 0.002,
    notional_usd: float = 1000.0,
    max_symbols: Optional[int] = None,
    cache_only: bool = False,
    backtester: Optional[BaseBacktester] = None,
    backtest_strat_processed_dir: Optional[Path] = None,
) -> MVPReport:
    """Train baseline CatBoost, generate out-of-sample signals, and validate via backtester."""
    news_path = NORMALIZED_NEWS_DIR / f"{provider_name}.parquet"
    news_df = pd.read_parquet(news_path)
    news_df["published_ts"] = pd.to_datetime(news_df["published_ts"], utc=True)

    bars_by_symbol = load_bars_for_news(
        news_df,
        backtest_strat_processed_dir=backtest_strat_processed_dir,
        max_symbols=max_symbols,
        cache_only=cache_only,
    )
    dataset_df = build_training_dataset(
        news_df,
        bars_by_symbol,
        entry_delay_minutes=entry_delay_minutes,
        horizon_minutes=horizon_minutes,
        bar_interval_minutes=5,
    )

    model = CatBoostTrendModel(
        feature_mode=feature_mode,  # type: ignore[arg-type]
        horizon_minutes=horizon_minutes,
    )
    metrics, test_df = model.fit_and_evaluate(dataset_df)
    model.save(name=f"catboost_{feature_mode}")

    signals_df = generate_signals(
        model,
        test_df,
        score_threshold=score_threshold,
        entry_delay_minutes=entry_delay_minutes,
    )
    save_signals(signals_df, filename=f"signals_{feature_mode}.parquet")

    bt = backtester or BacktestStratAdapter()
    results: List[BacktestRunResult] = []
    if not signals_df.empty:
        test_start = signals_df["signal_ts"].min() - pd.Timedelta(hours=1)
        for sym, sym_sigs in signals_df.groupby("symbol"):
            if (sym_sigs["direction"] == 0).all():
                continue
            bars = bars_by_symbol.get(sym)
            if bars is None or bars.empty:
                continue
            test_bars = bars[bars["bar_ts"] >= test_start]
            if test_bars.empty:
                continue
            results.append(
                bt.run(
                    symbol=sym,
                    bars=test_bars,
                    signals=sym_sigs,
                    notional_usd=notional_usd,
                    bar_interval_minutes=5,
                )
            )

    return MVPReport(
        news_count=len(news_df),
        symbols_count=len(bars_by_symbol),
        metrics=metrics,
        backtest_results=results,
    )


def run_rule_backtest_pipeline(
    provider_name: str = "binance",
    entry_delay_minutes: int = 5,
    horizon_minutes: int = 240,
    notional_usd: float = 1000.0,
    max_symbols: Optional[int] = None,
    cache_only: bool = True,
    backtester: Optional[BaseBacktester] = None,
    backtest_strat_processed_dir: Optional[Path] = None,
) -> List[BacktestRunResult]:
    """Run deterministic rule-based backtest (SHORT on listing/futures) without ML."""
    news_path = NORMALIZED_NEWS_DIR / f"{provider_name}.parquet"
    news_df = pd.read_parquet(news_path)
    news_df["published_ts"] = pd.to_datetime(news_df["published_ts"], utc=True)

    bars_by_symbol = load_bars_for_news(
        news_df,
        backtest_strat_processed_dir=backtest_strat_processed_dir,
        max_symbols=max_symbols,
        cache_only=cache_only,
    )
    matched_news = news_df[news_df["okx_symbol"].isin(bars_by_symbol)]
    signals_df = generate_rule_signals(
        matched_news,
        entry_delay_minutes=entry_delay_minutes,
        horizon_minutes=horizon_minutes,
    )
    bt = backtester or BacktestStratAdapter()
    results: List[BacktestRunResult] = []
    for sym, sym_sigs in signals_df.groupby("symbol"):
        bars = bars_by_symbol.get(sym)
        if bars is None or bars.empty:
            continue
        results.append(
            bt.run(
                symbol=sym,
                bars=bars,
                signals=sym_sigs,
                notional_usd=notional_usd,
                bar_interval_minutes=5,
            )
        )
    return results

