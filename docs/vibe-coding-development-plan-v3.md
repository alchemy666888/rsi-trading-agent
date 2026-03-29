# Vibe Coding Development Plan V3

## 1. Objective

This plan defines how to enhance the existing `rsi-trading-agent` codebase so the final agent behavior matches:

- [developing-rsi-trading-ai-agents.md](/Users/antee/Documents/projects/rsi-trading-agent/docs/developing-rsi-trading-ai-agents.md)
- all 9 mandatory requirements provided by the user

This is a **development plan only**. It does not modify source code in this step.

Target historical scope for all research, simulation, and reports:

- `BTCUSDT`
- `2023-01-01` to `2025-12-31`
- `15m`, `1h`, `4h`, `1d`

---

## 2. Current State Assessment (Existing Codebase)

Based on the current repository:

- Core runtime already exists in:
  - [main.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/main.py)
  - [agent.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/agent.py)
  - [tools.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/tools.py)
  - [research_agents.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/research_agents.py)
  - [memory.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/memory.py)
  - [planner.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/planner.py)
  - [reflection.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/reflection.py)

- Strengths already present:
  - Binance market data fetching exists via `ccxt`.
  - Multi-timeframe resampling exists (`15m`, `1h`, `4h`, `1d`).
  - Base indicators exist (RSI/MACD/MA/EMA/Bollinger).
  - News research module exists and writes output files.
  - Backtest report generation exists in markdown.
  - Memory and reflection loop exists across epochs.

- Main gaps versus required target:
  - Runtime still uses period-based fetch in core loop (`period="2y"`) instead of strict full-range window control.
  - Indicator set is incomplete relative to requirements (Ichimoku, VWAP, Volume Profile, richer features).
  - News layer needs explicit impact timing/duration modeling tied to price.
  - Simulation loop is epoch-style and weekly-style; required design is strict monthly recurring self-improvement walk-forward.
  - Need final strategy markdown artifact and final frozen backtest report flow.
  - Need stronger anti-leakage controls and richer acceptance metrics.

---

## 3. Delivery Principles

1. Align implementation to [developing-rsi-trading-ai-agents.md](/Users/antee/Documents/projects/rsi-trading-agent/docs/developing-rsi-trading-ai-agents.md) exactly.
2. Preserve and extend existing modules instead of rewriting the project from scratch.
3. Enforce deterministic historical windows and no forward data leakage.
4. Keep each phase testable with clear acceptance gates.
5. Build the monthly learning loop first, then strategy consolidation and final backtest.

---

## 4. Target Architecture (Post-Enhancement)

The enhanced codebase should operate as a coordinated pipeline with these logical components:

1. Orchestrator
2. Market Data Ingestion
3. Feature Engineering
4. News Intelligence
5. Regime Analyzer
6. Paper Trading Engine
7. Monthly Reflection and Learning
8. Strategy Authoring
9. Final Frozen Backtest and Reporting

Primary run mode:

- month-by-month walk-forward from `2023-01` through `2025-12`
- monthly update cycle:
  - trade month
  - evaluate month
  - learn lesson
  - update strategy
  - apply to next month

---

## 5. Work Breakdown Structure (WBS)

## 5.1 Phase A: Foundation Alignment and Config Hardening

Goal:

- make runtime windows explicit, reproducible, and aligned to requirements

Planned changes:

1. Add central project config for:
   - symbol = `BTCUSDT`
   - start = `2023-01-01`
   - end = `2025-12-31`
   - timeframes = `15m,1h,4h,1d`
   - monthly iteration mode
2. Replace ambiguous period-based defaults in orchestration with explicit date range calls.
3. Standardize timezone to UTC end-to-end.
4. Add run-manifest metadata for each run (parameters, model, data hashes, timestamps).

Primary files to enhance:

- [main.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/main.py)
- [agent.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/agent.py)
- [tools.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/tools.py)
- [README.md](/Users/antee/Documents/projects/rsi-trading-agent/README.md)

Acceptance gate:

- one command can launch a run that clearly states full-range configuration and monthly mode.

---

## 5.2 Phase B: Binance Multi-Timeframe Data Pipeline

Goal:

- reliably fetch complete BTC historical price and volume data for all required intervals

Planned changes:

1. Implement/rework ingestion flow to fetch and store:
   - `15m`
   - `1h`
   - `4h`
   - `1d`
2. Use free Binance endpoints via current `ccxt` integration.
3. Add pagination and retries with checkpoint resume.
4. Validate candle continuity per timeframe:
   - no missing bars
   - no duplicates
   - correct interval spacing
5. Persist canonical raw datasets with deterministic naming.

Primary files to enhance:

- [tools.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/tools.py)
- [research_agents.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/research_agents.py)
- [tests/test_tools.py](/Users/antee/Documents/projects/rsi-trading-agent/tests/test_tools.py)
- [tests/test_research_agents.py](/Users/antee/Documents/projects/rsi-trading-agent/tests/test_research_agents.py)

Data outputs:

- `data/btc_usdt_15m_2023-01-01_2025-12-31.csv`
- `data/btc_usdt_1h_2023-01-01_2025-12-31.csv`
- `data/btc_usdt_4h_2023-01-01_2025-12-31.csv`
- `data/btc_usdt_1d_2023-01-01_2025-12-31.csv`

Acceptance gate:

- all four timeframe files exist and pass continuity checks across full date range.

---

## 5.3 Phase C: Indicator and Feature Expansion

Goal:

- pre-calculate complete multi-timeframe indicators required by the spec

Planned changes:

1. Extend feature engine to include:
   - MA, EMA, SMA families
   - MACD and MACD histogram
   - RSI
   - Bollinger Bands and width
   - Ichimoku Cloud metrics
   - VWAP and rolling VWAP
   - Volume Profile approximations and value zones
2. Add supportive derived features:
   - trend alignment score
   - volatility regime flags
   - breakout confidence
   - volume imbalance features
3. Pre-calculate all features before simulation loop begins.
4. Store features per timeframe and month-partitioned slices for fast retrieval.

Primary files to enhance:

- [tools.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/tools.py)
- [research_agents.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/research_agents.py)
- [tests/test_tools.py](/Users/antee/Documents/projects/rsi-trading-agent/tests/test_tools.py)

Acceptance gate:

- feature tables include all required indicators for each timeframe and are queryable by timestamp.

---

## 5.4 Phase D: Historical News and Sentiment Impact Engine

Goal:

- ingest historical BTC/crypto/Web3 news and model price impact timing/duration

Planned changes:

1. Expand news ingestion for full period `2023-01-01` to `2025-12-31`.
2. Normalize and store:
   - `published_at`
   - source
   - title
   - summary/full-text when available
   - topic tags
3. Add sentiment scoring fields:
   - polarity
   - strength
   - confidence
4. Add event-impact analysis fields:
   - estimated lag-to-effect (`0-1h`, `1-4h`, `4-12h`, `12-24h`, `1-3d`, `3-7d`)
   - estimated duration of effect
   - dominant affected timeframe (`15m`, `1h`, `4h`, `1d`)
5. Persist enriched news dataset and analysis markdown/json.

Primary files to enhance:

- [research_agents.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/research_agents.py)
- [tools.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/tools.py)
- [tests/test_research_agents.py](/Users/antee/Documents/projects/rsi-trading-agent/tests/test_research_agents.py)

Data outputs:

- `news/btc_news_2023-01-01_2025-12-31.json`
- `news/btc_news_2023-01-01_2025-12-31_analysis.json`
- `news/btc_news_2023-01-01_2025-12-31_analysis.md`

Acceptance gate:

- news records contain sentiment and impact-window fields with non-empty monthly coverage.

---

## 5.5 Phase E: Multi-Timeframe Paper Trading Engine Upgrade

Goal:

- ensure paper trading logic follows required timeframe roles exactly

Planned changes:

1. Enforce strategy decision hierarchy:
   - use `1d` and `4h` for market regime and trend context
   - use `1h` and `15m` for entry and exit timing
2. Integrate inputs per trade:
   - historical price at decision time
   - pre-calculated indicators
   - news sentiment and active impact windows
   - regime and feature summaries
3. Add explicit no-trade state when signals conflict.
4. Upgrade trade logs with detailed rationale:
   - trend context
   - trigger confirmation
   - news filter state
   - risk controls applied
5. Keep cost model explicit:
   - fee
   - slippage
   - position/risk constraints

Primary files to enhance:

- [tools.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/tools.py)
- [agent.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/agent.py)
- [planner.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/planner.py)
- [tests/test_tools.py](/Users/antee/Documents/projects/rsi-trading-agent/tests/test_tools.py)

Acceptance gate:

- trade records clearly show `4h/1d` context plus `15m/1h` execution rationale for each entry/exit.

---

## 5.6 Phase F: Monthly Recurring Self-Learning Loop

Goal:

- migrate from generic epoch loop to strict monthly walk-forward self-improvement

Planned changes:

1. Replace epoch-first orchestration with monthly scheduler:
   - iterate months `2023-01` through `2025-12`
2. For each month:
   - load only data/news up to month end
   - run paper trading for that month
   - compute monthly metrics
   - generate lessons
   - apply constrained strategy updates for next month
3. Store monthly memory artifacts:
   - best patterns
   - worst patterns
   - strategy deltas
4. Add anti-overfitting guardrails:
   - bounded parameter changes
   - rationale required per parameter update
   - reject unstable changes

Primary files to enhance:

- [agent.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/agent.py)
- [memory.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/memory.py)
- [reflection.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/reflection.py)
- [planner.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/planner.py)
- [observability.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/observability.py)
- [tests/test_memory.py](/Users/antee/Documents/projects/rsi-trading-agent/tests/test_memory.py)
- [tests/test_planner_reflection.py](/Users/antee/Documents/projects/rsi-trading-agent/tests/test_planner_reflection.py)

Data model upgrade:

- extend SQLite schema to monthly strategy memory with:
  - month_id
  - strategy_json
  - metrics_json
  - score
  - lesson
  - change_log
  - validation_flags

Acceptance gate:

- the run proceeds month-by-month and strategy applied in month `N+1` is derived from month `N` lessons.

---

## 5.7 Phase G: Strategy Authoring Artifact

Goal:

- produce final learned strategy as markdown after monthly loop completes

Planned changes:

1. Create strategy synthesis step after final month.
2. Aggregate monthly winners/failures and robust rules.
3. Write final strategy markdown with:
   - market scope
   - indicator stack
   - regime logic
   - entry rules
   - exit rules
   - no-trade rules
   - risk controls
   - known weaknesses

Primary files to enhance:

- [agent.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/agent.py)
- [reflection.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/reflection.py)

Output target:

- `docs/final-btc-strategy.md`

Acceptance gate:

- final strategy markdown is generated automatically at end of full monthly run.

---

## 5.8 Phase H: Final Frozen Backtest and Detailed Report

Goal:

- run a final backtest with the frozen final strategy and produce detailed report

Planned changes:

1. Freeze final strategy parameters and logic.
2. Run full-period backtest using:
   - historical market data
   - historical news
   - pre-calculated indicators
   - pre-analyzed impact features
3. Expand report detail:
   - return metrics
   - risk metrics
   - monthly table
   - regime segmentation
   - long/short split
   - best/worst trades
   - streak behavior
   - narrative explanation

Primary files to enhance:

- [tools.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/tools.py)
- [agent.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/agent.py)
- [tests/test_tools.py](/Users/antee/Documents/projects/rsi-trading-agent/tests/test_tools.py)

Output targets:

- `backtest/final_strategy_backtest_report.md`
- optional run-stamped copies under `backtest/`

Acceptance gate:

- backtest report includes detailed metric blocks and explanatory narrative tied to final strategy.

---

## 6. Requirement-by-Requirement Implementation Mapping

1. Recurring self-learn/self-improve from historical data and news:
   - implemented by Phase F monthly walk-forward, memory persistence, and lesson-driven updates.
2. Binance free API BTC data `2023-01-01` to `2025-12-31` for `15m/1h/4h/1d`:
   - implemented by Phase B ingestion with explicit date windows and continuity validation.
3. Pre-calculated indicators including MA/EMA/SMA, MACD, RSI, Bollinger, Ichimoku, VWAP, Volume Profile:
   - implemented by Phase C feature expansion.
4. Historical BTC/crypto/Web3 news download + sentiment + effect timing/duration analysis:
   - implemented by Phase D news intelligence and impact-window modeling.
5. Paper trading with historical data/news/indicators/analysis + detailed report:
   - implemented by Phase E trade engine and report upgrades.
6. Use `4h`/`1d` for trend and `15m`/`1h` for entry/exit:
   - implemented as strict decision hierarchy in Phase E.
7. Monthly recurring paper trade -> summarize -> learn -> self-improve -> next month:
   - implemented directly by Phase F orchestration and memory schema.
8. Final strategy markdown generation:
   - implemented by Phase G strategy authoring stage.
9. Final strategy backtest + detailed backtest report:
   - implemented by Phase H frozen final backtest stage.

---

## 7. Proposed File-Level Enhancement Plan

1. [main.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/main.py)
   - switch entrypoint to monthly walk-forward orchestrator
   - wire final strategy and final backtest generation
2. [agent.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/agent.py)
   - refactor run loop from epoch-based to month-based
   - integrate strategy update lifecycle
3. [tools.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/tools.py)
   - expand indicators and impact-aware simulation tooling
   - add detailed reporting utilities
4. [research_agents.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/research_agents.py)
   - full-range multi-timeframe data artifacts
   - richer news intelligence fields
5. [memory.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/memory.py)
   - monthly memory schema and retrieval policies
6. [planner.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/planner.py)
   - parameter policy for constrained monthly updates
7. [reflection.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/reflection.py)
   - structured monthly lessons and update recommendations
8. [observability.py](/Users/antee/Documents/projects/rsi-trading-agent/btc_self_improve_agent/observability.py)
   - month-level traces and run manifests
9. [tests/test_tools.py](/Users/antee/Documents/projects/rsi-trading-agent/tests/test_tools.py)
   - coverage for new indicators and final report metrics
10. [tests/test_research_agents.py](/Users/antee/Documents/projects/rsi-trading-agent/tests/test_research_agents.py)
   - data/news schema validation and date-range integrity checks
11. [tests/test_memory.py](/Users/antee/Documents/projects/rsi-trading-agent/tests/test_memory.py)
   - monthly memory retrieval and update-history tests
12. [tests/test_planner_reflection.py](/Users/antee/Documents/projects/rsi-trading-agent/tests/test_planner_reflection.py)
   - planner/reflection structured outputs for monthly evolution
13. [README.md](/Users/antee/Documents/projects/rsi-trading-agent/README.md)
   - update usage docs to new monthly self-improving flow

---

## 8. Data Contracts and Artifacts

## 8.1 Market Data Contract

Per candle fields:

- timestamp/open_time
- open
- high
- low
- close
- volume
- close_time
- quote_asset_volume
- number_of_trades
- taker_buy_base_asset_volume
- taker_buy_quote_asset_volume
- timeframe

## 8.2 Indicator Contract

Per timeframe row:

- base OHLCV columns
- MA/SMA/EMA families
- RSI
- MACD + signal + histogram
- Bollinger bands + width
- Ichimoku components
- VWAP features
- Volume Profile zone flags
- regime/helper derived features

## 8.3 News Contract

Per item:

- published_at
- source
- title
- summary/text
- tags/topics
- sentiment_polarity
- sentiment_strength
- sentiment_confidence
- impact_lag_bucket
- impact_duration_bucket
- impacted_timeframe

## 8.4 Trade Log Contract

Per trade:

- trade_id
- side
- entry_time
- exit_time
- entry_price
- exit_price
- size
- pnl_pct
- return_pct
- bars_held
- higher_tf_context_summary (`4h/1d`)
- lower_tf_trigger_summary (`15m/1h`)
- news_filter_summary
- entry_rationale
- exit_rationale

---

## 9. Testing and Validation Plan

## 9.1 Unit Tests

1. Indicator calculations:
   - verify all required indicators exist and handle edge windows safely.
2. Data ingestion:
   - verify pagination and continuity checks.
3. News enrichment:
   - verify sentiment and impact fields.
4. Monthly memory:
   - verify write/read for month-level lessons and strategy diffs.
5. Report generation:
   - verify detailed sections and key metrics are present.

## 9.2 Integration Tests

1. End-to-end monthly loop on shortened sample range.
2. Full timeframe interaction checks for `1d/4h` context and `1h/15m` execution.
3. Final strategy artifact generation.
4. Final frozen backtest report generation.

## 9.3 Leakage Prevention Tests

1. Ensure no future news or future candles are accessible at each monthly step.
2. Ensure strategy updates at month `N+1` use only lessons from `<= N`.

---

## 10. Observability and Auditability

Add trace outputs for each month:

- month id
- strategy version id
- key metrics
- lesson summary
- parameter changes
- validation flags
- artifact paths

Store:

- `traces/month_YYYY-MM.json`
- run manifest files with config and data versions

This enables reproducible analysis of why strategy changed over time.

---

## 11. Suggested Milestones and Sequence

1. Milestone 1:
   - Phase A + Phase B complete
   - full four-timeframe market dataset validated
2. Milestone 2:
   - Phase C + Phase D complete
   - indicators and news-impact intelligence ready
3. Milestone 3:
   - Phase E complete
   - paper trading engine fully aligned with required timeframe roles
4. Milestone 4:
   - Phase F complete
   - monthly self-improvement loop operational
5. Milestone 5:
   - Phase G + Phase H complete
   - final strategy markdown and final backtest report generated

---

## 12. Risks and Mitigations

1. Risk:
   - news source gaps or inconsistent historical coverage
   - Mitigation: use multiple source windows and mark coverage confidence.
2. Risk:
   - overfitting due to aggressive monthly tuning
   - Mitigation: bounded parameter change policy and robustness checks.
3. Risk:
   - simulation realism drift
   - Mitigation: explicit cost/slippage assumptions and transparent reporting.
4. Risk:
   - data-quality issues in high-frequency candles
   - Mitigation: strict continuity checks and anomaly flags.
5. Risk:
   - LLM output inconsistency in planner/reflection
   - Mitigation: schema validation, sanitization, and deterministic fallbacks.

---

## 13. Definition of Done (DoD)

The enhancement effort is complete only when all conditions below are true:

1. Agent executes monthly walk-forward self-improvement from `2023-01` to `2025-12`.
2. Binance free API data is ingested for `15m`, `1h`, `4h`, `1d` across full date range.
3. Full required indicator family is pre-calculated and stored.
4. Historical BTC/crypto/Web3 news is ingested and enriched with sentiment plus impact timing/duration.
5. Paper trading uses historical data + news + indicators + analysis outputs and produces detailed monthly reports.
6. Runtime strictly applies `4h/1d` for market context and `15m/1h` for entry/exit.
7. Monthly lessons are persisted, strategy is updated, and next month uses updated strategy.
8. Final strategy markdown file is produced automatically.
9. Final frozen backtest is run and detailed backtest report is produced automatically.
10. Automated tests cover new core flows and pass.

---

## 14. Final Notes

This V3 plan upgrades the project from a good prototype into a complete self-improving quant research workflow. It keeps the existing codebase foundation, aligns directly with [developing-rsi-trading-ai-agents.md](/Users/antee/Documents/projects/rsi-trading-agent/docs/developing-rsi-trading-ai-agents.md), and provides a practical path to deliver all 9 requirements exactly.
