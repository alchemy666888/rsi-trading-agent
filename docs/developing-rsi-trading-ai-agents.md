# Developing RSI Trading AI Agents: Current BTC Strategy Design and Runtime Logic

## 1. Purpose

This document describes the current implemented design of the BTC self-improving trading agent in this repository. It replaces the earlier aspirational blueprint with a code-aligned explanation of how the system now works.

The implementation is built around four ideas:

1. use `1d` for slow macro bias,
2. use `4h` for directional confirmation,
3. use `1h` to confirm the setup shape,
4. use `15m` to trigger and manage execution.

The system is intentionally long-biased for BTC unless the daily regime is decisively bearish. News is treated as a risk modifier and cooldown filter, not as a standalone trade trigger.

## 2. Code Map

The main runtime is split across these modules:

- `btc_self_improve_agent/agent.py`: monthly walk-forward orchestration, warmup windows, monthly update gating, final report generation.
- `btc_self_improve_agent/planner.py`: default strategy parameters, parameter sanitization, LLM-guided strategy planning, conservative rule-based updates.
- `btc_self_improve_agent/tools.py`: indicator calculation, news enrichment, multi-timeframe signal generation, position simulation, metrics, reports.
- `btc_self_improve_agent/reflection.py`: monthly lessons and deterministic overrides when evidence is sparse or weak.

## 3. Data Inputs and Preprocessing

### 3.1 Market data

The runtime expects BTC OHLCV data for these timeframes:

- `15m`
- `1h`
- `4h`
- `1d`

Each frame is coerced to a UTC datetime index, sorted, deduplicated, and gap-filled. Missing OHLCV gaps are forward-filled conservatively so indicator windows remain stable.

### 3.2 Timestamp handling

After coercion, each frame index is shifted forward by exactly one bar length. This aligns each row to candle close time, which reduces lookahead risk when lower and higher timeframes are joined.

### 3.3 Monthly walk-forward windows

Monthly simulation does not trade on a cold start. For each month, the agent loads:

- the target month,
- a `365` day market-data warmup window before that month,
- a `7` day news lookback before that month.

Trading metrics are then evaluated only inside the requested monthly evaluation window.

## 4. Indicator and Feature Layer

Indicators are calculated per timeframe using strategy parameters where available.

### 4.1 Core indicators

The current feature layer includes:

- EMA short and EMA long
- MACD, MACD signal, MACD histogram
- RSI and fast RSI
- SMA `10`, `20`, `50`, `200`
- parameterized SMA short and SMA long
- Bollinger mid, upper, lower, width
- Ichimoku tenkan, kijun, span A, span B, lagging span, cloud bias
- VWAP and rolling VWAP
- volume profile approximations: POC, VAL, VAH
- ATR and ATR percentage
- volume surge ratio
- body-to-range ratio
- trend slope features
- trend alignment score

### 4.2 Feature intent by timeframe

The implemented logic uses the stack as follows:

- `1d`: slow bull or bear classification.
- `4h`: directional confirmation and regime reinforcement.
- `1h`: setup confirmation, especially breakout or breakdown context.
- `15m`: discrete trigger event and tactical execution timing.

## 5. News Intelligence Layer

### 5.1 Enrichment

Historical news is enriched with:

- heuristic sentiment score
- bullish, bearish, or neutral polarity
- sentiment strength
- event type inference such as `etf`, `regulation`, `macro`, `hack`, `liquidation`, `mining`, `stablecoin`
- impact lag bucket
- impact duration bucket
- impacted timeframe hint
- publication timestamp confidence

### 5.2 Confidence-aware impact

If publication time quality is weak, the system reduces the effect of that news item. If publication time is missing, the record is still handled deterministically but marked low confidence.

### 5.3 News context used during trading

The runtime converts enriched news into per-bar context:

- `impact`: signed directional influence
- `active_count`: number of active events
- `hard_cooldown`: blocks entry after major, high-confidence events
- `soft_cooldown`: scales down new entries in medium-confidence event windows

This is intentionally asymmetric:

- favorable news can only boost size modestly,
- adverse news can reduce size more strongly,
- strong event windows can block fresh entries altogether.

News does not create a trade by itself.

## 6. Multi-Timeframe Regime Logic

The live backtest path is `_run_multi_timeframe_backtest()` in `btc_self_improve_agent/tools.py`.

### 6.1 Daily macro bias

The daily regime is built from a score that blends:

- close versus `SMA_200`
- close versus `SMA_long`
- EMA short versus EMA long
- MACD versus MACD signal
- Ichimoku cloud bias
- 20-bar trend slope

From that score the engine derives:

- `macro_bull`
- `macro_bear`
- `trend_1d` in `{1, 0, -1}`

The daily filter is intentionally slower than the earlier version so BTC is not classified bearish too easily during ordinary corrections.

### 6.2 4h confirmation

The `4h` confirmation score blends:

- close versus EMA short
- close versus SMA long
- close versus rolling VWAP
- MACD versus MACD signal
- Ichimoku cloud bias

This becomes:

- `trend_4h_bull_score`
- `trend_4h_bear_score`
- `trend_4h` in `{1, 0, -1}`

### 6.3 Higher-timeframe alignment

The engine computes:

- `higher_aligned`: `1d` and `4h` point the same way
- `higher_long_score`
- `higher_short_score`

The active regimes are then:

- `bull_regime = macro_bull and 4h bull confirmation`
- `bear_regime = macro_bear and 4h bear confirmation`
- otherwise `range`

## 7. Entry Logic

### 7.1 Current live setup families

The current production entry path is intentionally narrow.

Active setup families:

- `long_breakout`
- `short_breakdown`

Inactive in the live entry path right now:

- long pullback
- long trend continuation
- short rally fade
- short trend continuation

Some related parameters are still present in the strategy schema for compatibility, but they are not used by the current signal engine.

### 7.2 Long-side philosophy

Longs are allowed only when all of these are true:

- `bull_regime` is active
- `1d` and `4h` are aligned
- higher-timeframe long score clears the minimum quality threshold

This encodes the intended BTC long bias.

### 7.3 Short-side philosophy

Shorts are allowed only when all of these are true:

- `bear_regime` is active
- `1d` and `4h` are aligned
- daily bear score is very strong

Shorts are deliberately more restrictive than longs.

### 7.4 1h setup confirmation

A long breakout needs `1h` confirmation such as:

- close at or above the rolling `1h` breakout high
- `1h` volume surge
- `1h` MACD confirmation
- `1h` price above rolling VWAP

A short breakdown mirrors that logic on the downside and also requires the `1h` close to stay below `SMA_200`.

### 7.5 15m trigger

A long trigger then requires a discrete `15m` event:

- prior `15m` close below the rolling `15m` breakout high
- current `15m` close at or above that breakout high
- `15m` volume surge
- `15m` MACD confirmation
- improving `15m` MACD histogram
- `15m` price above rolling VWAP

A short trigger mirrors that structure on the downside.

### 7.6 Discrete entries, not continuous activation

The tuned implementation only fires on transition bars. It does not keep generating fresh entries on every bar that happens to remain above a threshold.

That change was important because the older score-combination approach overtraded badly.

### 7.7 Entry workflow field

When a trade candidate survives the full stack, the runtime records:

- `entry_signal_tf = 1h+15m`
- `setup_family = long_breakout` or `short_breakdown`

This is what the reports now display.

## 8. Position Sizing and Risk Management

The execution simulator runs on the `15m` base frame.

### 8.1 Sizing

Position size is capped by:

- `max_position`
- raw signal magnitude
- a risk cap based on `max_loss_per_trade / ATR_distance`

This keeps trades from growing too large in high-volatility conditions.

### 8.2 Stops and targets

The simulator uses:

- ATR-based initial stop logic
- ATR-based target logic
- breakeven arming at `breakeven_r_multiple`
- trailing stop updates as price moves
- setup-aware target and hold adjustments

Because only `long_breakout` and `short_breakdown` are currently active, the setup-specific behavior that matters most is:

- `long_breakout`: longer hold allowance and slightly larger target multiple
- `short_breakdown`: slightly reduced target multiple and shorter hold than the long breakout case

### 8.3 Exit reasons

The runtime records explicit exit events:

- `stop_loss_hit`
- `take_profit_hit`
- `higher_timeframe_invalidation`
- `signal_flip`
- `max_holding_reached`

These reasons feed directly into trade logs and backtest reports.

### 8.4 Cooldowns

There are two cooldown mechanisms:

- `cooldown_bars_after_news`: event-driven hard or soft entry suppression
- `trade_cooldown_bars`: pause after exits before a new position may open

## 9. Monthly Self-Improvement Loop

The orchestrator in `agent.py` runs month by month across the configured history.

### 9.1 Monthly sequence

For each month the agent:

1. slices warmup data and month data,
2. filters news with a short pre-month lookback,
3. runs the backtest only on information available at that time,
4. scores the month,
5. reflects on failures and successes,
6. updates the strategy only if enough evidence exists,
7. stores reports, trace data, and memory.

### 9.2 Update gating

The system no longer updates parameters after any weak single month.

A monthly parameter update requires aggregated evidence across the current month plus the previous two months:

- at least `2` active months,
- at least `12` trades in aggregate.

If evidence is too thin, the strategy is held steady and the skip reason is stored.

### 9.3 Reflection overrides

`reflection.py` now uses deterministic guardrails when LLM feedback would be too noisy:

- `0` trades: relax only trigger quality modestly, keep macro filter intact, keep shorts restrictive
- `< 5` trades: avoid major changes, prefer stability over optimization
- losing month with weak trade quality: reduce countertrend shorts, prefer breakout or continuation over shallow pullbacks, reduce news amplification

### 9.4 Rule-based parameter drift control

`planner.py` now nudges parameters in safer directions:

- weak performance reduces `news_weight`
- weak performance raises `short_score_threshold`
- weak performance can extend `trade_cooldown_bars`
- low-trade months can relax breakout filters only modestly
- the update logic avoids the earlier drift toward globally compressed RSI thresholds and excessive news amplification

## 10. Reporting and Metrics

The backtest output now includes trade-level and portfolio-level metrics such as:

- total return
- annualized return
- Sharpe ratio
- Sortino ratio
- maximum drawdown
- Calmar ratio
- trade-level win rate
- profit factor
- expectancy
- average win
- average loss
- average holding bars
- trade count
- monthly breakdown
- regime breakdown
- narrative summary

### 10.1 Reliability flags

The engine also attaches caution flags such as:

- `low_sample_size`
- `one_sided_exposure`
- `extreme_win_rate_small_sample`
- `profit_factor_unstable`
- `negative_sharpe`
- `profit_factor_below_one`
- `negative_total_return`

These flags are written into monthly and final reports so poor-looking metrics are not overinterpreted.

### 10.2 Trade rationale text

Trade logs now explain entries and exits in a way that matches the current engine:

- higher-timeframe rationale references `1d` macro bias and `4h` confirmation
- lower-timeframe rationale references the `1h + 15m` workflow and setup family
- news rationale references active event count, directional impact, and cooldown state

## 11. Current Default Strategy Profile

The current default strategy in `planner.py` is tuned around a sparse breakout-led BTC workflow.

Key defaults:

- `entry_timeframe: auto`
- `max_position: 0.55`
- `stop_atr_multiple: 2.6`
- `take_profit_atr_multiple: 5.0`
- `trend_filter_strength: 0.62`
- `long_score_threshold: 0.82`
- `short_score_threshold: 0.95`
- `score_hysteresis: 0.08`
- `breakout_volume_surge: 1.20`
- `short_breakdown_volume_surge: 1.60`
- `trade_cooldown_bars: 32`
- `breakout_1h_window: 72`
- `breakout_15m_window: 32`
- `news_weight: 0.12`
- `news_impact_cap: 0.6`

This profile is intentionally conservative and much less reactive than the earlier version.

## 12. What Changed in This Tuning Pass

This revision addressed the main issues observed in prior review and backtest output:

- enforced the intended `1d -> 4h -> 1h -> 15m` hierarchy
- removed the old effective "pick one entry timeframe" behavior
- made BTC structurally long-biased and kept shorts highly selective
- shifted the live signal engine to discrete breakout and breakdown events
- reduced news from alpha amplifier to risk modifier
- slowed monthly updates so thin evidence does not drift parameters aggressively
- improved reporting so rationales, reliability flags, and trade-level metrics match runtime behavior

## 13. Known Limitations

The current implementation is materially cleaner than the earlier local baseline, but it is not finished research.

Important limitations:

- the live entry engine currently activates only breakout-led setups
- some legacy strategy fields remain in the schema for compatibility even though the live signal path does not use them yet
- the strategy still needs more research before it can be considered robust across all BTC environments
- validation in this repository is currently stronger through direct simulation and report inspection than through a fully configured test suite

## 14. Summary

The repository now implements a narrower, more disciplined BTC trading agent than the earlier design document described.

In practice, the current system is:

- a multi-timeframe BTC backtester with `1d` macro bias and `4h` confirmation,
- a breakout-led execution model using `1h` setup confirmation and `15m` trigger timing,
- a confidence-aware news risk filter,
- a monthly walk-forward learner with evidence gating,
- and a reporting pipeline that records not just returns, but why the system behaved as it did.

That is the latest code-aligned strategy design for this project.
