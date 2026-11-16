# Domain docs

## Layout

This is a single-context repository (`binance_news_price_prediction`) for predicting LFT/MFT crypto price trends from news (starting with Binance announcements) and validating signals on OKX market data via `backtest_strat`:

- `README.md` at the root: project goal, models, and server execution plan.
- `CONTEXT.md` at the root: domain terminology and context (e.g., news providers, `published_ts` alignment, event types, LFT/MFT horizons).
- `docs/adr/`: architectural decisions (including integration with `backtest_strat` and the pluggable backtester contract).

## Consumer rules

Before exploring the codebase, read `README.md`, `CONTEXT.md`, and ADRs relevant to
the area being changed.

If these files do not exist, proceed silently. Do not suggest creating
them upfront. The domain-modeling skill creates them lazily when
terminology or decisions are resolved.

Use the vocabulary defined in `CONTEXT.md` in issues, proposals,
hypotheses, and tests. Reconsider unfamiliar terminology or note a
real glossary gap for domain-modeling.

If a proposal contradicts an ADR, explicitly identify the ADR and
explain why its decision should be reconsidered.
