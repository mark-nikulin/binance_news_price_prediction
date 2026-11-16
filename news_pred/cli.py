"""CLI entry point for news_pred."""

import argparse
import logging

from news_pred.backtest.contract import compute_portfolio_sharpe
from news_pred.pipeline import (
    ingest_and_normalize,
    run_event_study_pipeline,
    run_mvp_pipeline,
    run_rule_backtest_pipeline,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(prog="np-cli", description="Binance News Price Prediction MVP")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Fetch and normalize news from provider")
    p_ing.add_argument("--provider", default="binance")
    p_ing.add_argument("--pages", type=int, default=5)
    p_ing.add_argument("--page-size", type=int, default=20)

    p_es = sub.add_parser("event-study", help="Run post-announcement drift Event Study")
    p_es.add_argument("--provider", default="binance")
    p_es.add_argument("--delay-min", type=int, default=5)
    p_es.add_argument("--max-symbols", type=int, default=None)

    p_mvp = sub.add_parser("run-mvp", help="Train CatBoost baseline, infer signals, and run backtest")
    p_mvp.add_argument("--provider", default="binance")
    p_mvp.add_argument("--mode", choices=["market_only", "market_plus_news", "market_plus_news_text"], default="market_plus_news")
    p_mvp.add_argument("--delay-min", type=int, default=5)
    p_mvp.add_argument("--horizon-min", type=int, default=240)
    p_mvp.add_argument("--threshold", type=float, default=0.002)
    p_mvp.add_argument("--notional-usd", type=float, default=1000.0)
    p_mvp.add_argument("--max-symbols", type=int, default=None)
    p_mvp.add_argument("--cache-only", action="store_true")

    p_rule = sub.add_parser("rule-backtest", help="Run zero-ML rule backtest (SHORT listing/futures)")
    p_rule.add_argument("--provider", default="binance")
    p_rule.add_argument("--delay-min", type=int, default=5)
    p_rule.add_argument("--horizon-min", type=int, default=240)
    p_rule.add_argument("--notional-usd", type=float, default=1000.0)
    p_rule.add_argument("--max-symbols", type=int, default=50)
    p_rule.add_argument("--cache-only", action="store_true")

    args = parser.parse_args()

    if args.cmd == "ingest":
        df = ingest_and_normalize(args.provider, max_pages=args.pages, page_size=args.page_size)
        print(f"Normalized {len(df)} rows across {df['okx_symbol'].nunique() if not df.empty else 0} symbols.")
    elif args.cmd == "event-study":
        summary = run_event_study_pipeline(
            provider_name=args.provider,
            entry_delay_minutes=args.delay_min,
            max_symbols=args.max_symbols,
        )
        print(summary.to_string(index=False) if not summary.empty else "No matching market data.")
    elif args.cmd == "run-mvp":
        report = run_mvp_pipeline(
            provider_name=args.provider,
            feature_mode=args.mode,
            entry_delay_minutes=args.delay_min,
            horizon_minutes=args.horizon_min,
            score_threshold=args.threshold,
            notional_usd=args.notional_usd,
            max_symbols=args.max_symbols,
            cache_only=args.cache_only,
        )
        print("Train/Test Metrics:", report.metrics)
        total_pnl = sum(r.net_pnl for r in report.backtest_results)
        total_fees = sum(r.fees for r in report.backtest_results)
        total_fills = sum(r.fills_count for r in report.backtest_results)
        port_sharpe = compute_portfolio_sharpe(report.backtest_results)
        for r in report.backtest_results:
            print(f"  {r.symbol}: net_pnl=${r.net_pnl:.2f}, fees=${r.fees:.2f}, fills={r.fills_count}, max_dd=${r.max_drawdown:.2f}, sharpe={r.sharpe_ratio:.2f}")
        print(f"Portfolio Total ({len(report.backtest_results)} symbols, ${args.notional_usd:.0f}/trade): net_pnl=${total_pnl:.2f}, fees=${total_fees:.2f}, fills={total_fills}, sharpe={port_sharpe:.2f}")
    elif args.cmd == "rule-backtest":
        results = run_rule_backtest_pipeline(
            provider_name=args.provider,
            entry_delay_minutes=args.delay_min,
            horizon_minutes=args.horizon_min,
            notional_usd=args.notional_usd,
            max_symbols=args.max_symbols,
            cache_only=args.cache_only,
        )
        total_pnl = sum(r.net_pnl for r in results)
        total_fees = sum(r.fees for r in results)
        total_fills = sum(r.fills_count for r in results)
        port_sharpe = compute_portfolio_sharpe(results)
        for r in results:
            print(f"  {r.symbol}: net_pnl=${r.net_pnl:.2f}, fees=${r.fees:.2f}, fills={r.fills_count}, max_dd=${r.max_drawdown:.2f}, sharpe={r.sharpe_ratio:.2f}")
        print(f"Rule Portfolio Total ({len(results)} symbols, ${args.notional_usd:.0f}/trade): net_pnl=${total_pnl:.2f}, fees=${total_fees:.2f}, fills={total_fills}, sharpe={port_sharpe:.2f}")


if __name__ == "__main__":
    main()

