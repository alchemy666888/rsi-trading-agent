# Developing RSI Trading AI Agents: A Self-Improving BTC Paper-Trading Blueprint

## 1. Mission and Scope

This document upgrades the earlier generic AI-agent design into a concrete blueprint for a **recurring self-learning and self-improving BTC trading agent**. The target system is not a chatbot and not a one-off backtest script. It is a research-and-execution agent system that repeatedly:

1. downloads and organizes historical BTC market data and historical crypto news,
2. pre-calculates multi-timeframe indicators and market structure features,
3. analyzes news sentiment and news-to-price impact duration,
4. performs historical paper trading with strict multi-timeframe logic,
5. evaluates monthly performance and extracts lessons,
6. updates its strategy rules and decision policy for the next month,
7. converges toward a final strategy,
8. documents that final strategy in markdown, and
9. runs a full historical backtest with a detailed backtest report.

The full research period is:

- **Market data range**: `2023-01-01` to `2025-12-31`
- **Primary asset**: `BTCUSDT`
- **Execution style**: historical paper trading and backtesting only
- **Required timeframes**: `15m`, `1h`, `4h`, `1d`
- **Higher-timeframe context**: `4h` and `1d`
- **Lower-timeframe entry/exit**: `15m` and `1h`

This document is intentionally written as a system design and operating blueprint so it can guide implementation later without changing the current codebase now.

---

## 2. Core Agent Foundations for Quant Trading

### 2.1 AI Agent Core Structure

An effective trading AI agent still follows the classical agent model:

- **Brain**: LLM reasoning layer for planning, synthesis, reflection, hypothesis generation, and strategy revision.
- **Environment**: historical market data, historical news corpus, indicator store, paper-trading simulator, report outputs, and strategy memory.
- **Sensors**: Binance market-data API responses, news APIs or historical news archives, computed indicators, event calendars, prior-month performance metrics, and risk summaries.
- **Actuators**: data ingestion jobs, feature calculators, sentiment analyzers, trade simulators, report generators, and strategy-updating workflows.
- **Loop**: perceive -> reason -> act -> evaluate -> learn -> update -> repeat.

### 2.2 Trading-Specific Agent Loop

For this project, the loop is:

1. **Perceive**
   Load BTC OHLCV data, historical news, indicator tables, prior lessons, and current month trading context.
2. **Reason**
   Interpret higher-timeframe trend, lower-timeframe entry conditions, news regime, volatility regime, and risk constraints.
3. **Act**
   Execute paper trades on historical data using only information available up to each historical timestamp.
4. **Evaluate**
   Score trades and monthly performance using return, Sharpe, drawdown, expectancy, win/loss structure, regime behavior, and news-impact quality.
5. **Learn**
   Extract lessons such as “trend filter too loose,” “news shock requires longer cooldown,” or “RSI mean-reversion failed in breakout regimes.”
6. **Improve**
   Update strategy parameters, rule weights, regime filters, news-handling logic, risk rules, and entry/exit thresholds for the next month.
7. **Repeat**
   Roll the updated strategy into the next historical month and continue until the full dataset is completed.

This is the essential mechanism that fulfills the requirement for **recurring self-learning and self-improvement**.

---

## 3. Target System Architecture

The upgraded agent should be designed as a **multi-agent research-and-trading workflow**, not a single monolithic agent.

### 3.1 Recommended Agent Roles

- **Orchestrator Agent**
  Controls the monthly workflow, delegates tasks, enforces sequencing, and maintains run state.
- **Market Data Agent**
  Downloads BTC historical OHLCV and volume data from Binance, validates completeness, and standardizes timestamps.
- **Feature Engineering Agent**
  Pre-calculates indicators and market-structure features for all required timeframes.
- **News Intelligence Agent**
  Downloads or ingests historical BTC/crypto/Web3 news, tags themes, scores sentiment, and estimates price-impact lag and duration.
- **Regime Analysis Agent**
  Determines bull/bear/range conditions, volatility regimes, trend strength, and cross-timeframe alignment.
- **Paper Trading Agent**
  Simulates decisions candle by candle using only historically available data.
- **Reflection and Learning Agent**
  Reviews monthly outcomes, identifies failure patterns, and proposes strategy updates.
- **Strategy Authoring Agent**
  Consolidates the best learned policy into a final markdown trading strategy.
- **Backtest Report Agent**
  Runs the final strategy across the full period and writes a detailed backtest report.

### 3.2 Why Multi-Agent Matters

This separation is important because the system must do three different kinds of work well:

- deterministic data processing,
- probabilistic reasoning and synthesis,
- iterative self-improvement.

A specialized architecture reduces hallucination, improves observability, and makes monthly learning auditable.

---

## 4. Historical Market Data Requirement: Binance Free API

### 4.1 Required Market Dataset

The agent must be able to call the **Binance free API** and download BTC historical price and volume data for:

- **Symbol**: `BTCUSDT`
- **Date range**: `2023-01-01 00:00:00 UTC` to `2025-12-31 23:59:59 UTC`
- **Intervals**:
  - `15m`
  - `1h`
  - `4h`
  - `1d`

### 4.2 Required Fields

For each candle, the dataset should include at minimum:

- open time
- open
- high
- low
- close
- volume
- close time
- quote asset volume
- number of trades
- taker buy base asset volume
- taker buy quote asset volume

### 4.3 Data Quality Rules

The Market Data Agent should enforce:

- no missing candles inside the target range,
- consistent timezone handling, ideally normalized to UTC internally,
- no duplicate rows,
- exact interval continuity,
- validation that resampled and directly downloaded interval data align,
- metadata logging for source, request time, completeness, and checksum.

### 4.4 Why Direct Multi-Timeframe Data Matters

Even if resampling is possible, the design should retain explicit timeframe-aware datasets because:

- `1d` and `4h` define market context and directional bias,
- `1h` and `15m` define tactical entries and exits,
- feature timing and indicator lag can differ materially by timeframe,
- news impact often appears first on lower timeframes but confirms on higher ones.

---

## 5. Indicator and Feature Pre-Calculation Layer

### 5.1 Indicator Philosophy

The agent should **pre-calculate indicators before paper trading**, not compute them ad hoc during every reasoning step. This improves consistency, reproducibility, speed, and auditability.

### 5.2 Required Indicator Families

The system should compute indicators across the required timeframes as applicable:

- **Trend and smoothing**
  - SMA
  - EMA
  - MA variants across multiple lookbacks
- **Momentum**
  - RSI
  - MACD
  - MACD histogram
  - rate of change
- **Volatility**
  - Bollinger Bands
  - band width
  - ATR
- **Structure and trend state**
  - Ichimoku Cloud
  - cloud position
  - Tenkan/Kijun alignment
  - future cloud bias
- **Volume and participation**
  - VWAP
  - rolling VWAP
  - volume surge ratios
  - taker-buy imbalance
- **Market profile / auction context**
  - Volume Profile
  - high-volume node zones
  - low-volume node zones
  - value area high / low
- **Supportive context features**
  - swing high / low markers
  - trend slope
  - candle body-to-range ratio
  - gap and breakout flags
  - realized volatility

### 5.3 Timeframe-Aware Feature Design

Indicators should not be used identically across all timeframes.

- **1d**
  Macro trend, long directional bias, structural bull/bear/range regime.
- **4h**
  Intermediate regime, pullback quality, trend continuation vs exhaustion.
- **1h**
  Trade setup formation, confirmation, momentum alignment, stop placement logic.
- **15m**
  Fine entry timing, trigger candle, micro-structure, and exit refinement.

### 5.4 Derived Composite Features

Beyond raw indicators, the Feature Engineering Agent should create composite signals such as:

- higher-timeframe trend alignment score,
- lower-timeframe trigger quality score,
- trend-versus-mean-reversion regime classifier,
- volatility-adjusted RSI state,
- MACD momentum acceleration score,
- volume-confirmed breakout score,
- news-sensitive risk regime score,
- entry timing confidence score.

These derived features are especially valuable for monthly self-improvement because they are easier to compare, rank, and refine than raw indicators alone.

---

## 6. Historical News Intelligence Layer

### 6.1 News Scope

The system must download or otherwise ingest historical news related to:

- BTC / Bitcoin,
- cryptocurrency markets,
- Web3,
- macro news with strong BTC relevance,
- exchange, ETF, regulation, mining, stablecoin, and risk-event topics.

Target date range:

- **`2023-01-01` to `2025-12-31`**

### 6.2 News Data Expectations

Each news item should ideally include:

- publication timestamp,
- headline,
- full text or summary,
- source,
- URL or reference,
- category or topic tags,
- entities mentioned,
- region if relevant,
- confidence in parsing quality.

### 6.3 News Analysis Requirements

The News Intelligence Agent should analyze:

- sentiment polarity: bullish / bearish / neutral,
- sentiment strength,
- event type: regulation, ETF, macro, hack, adoption, liquidity, etc.,
- novelty versus repeated story,
- credibility of source,
- expected market relevance,
- expected impact lag,
- expected impact duration.

### 6.4 News-to-Price Impact Study

This project specifically requires understanding **how** and **how long** news affects BTC price. The agent should therefore estimate:

- whether the news effect appears immediately, within hours, or over multiple days,
- whether the effect is strongest on `15m`, `1h`, `4h`, or `1d`,
- whether the effect causes reversal, continuation, breakout, or volatility expansion,
- whether the effect decays quickly or persists,
- whether the effect depends on prior market regime.

### 6.5 Suggested Impact Windows

The system should evaluate each news item across multiple post-publication windows, such as:

- `0-1h`
- `1-4h`
- `4-12h`
- `12-24h`
- `1-3d`
- `3-7d`

This makes the agent capable of learning statements like:

- ETF approval headlines often create immediate momentum plus a secondary reaction window,
- regulatory fear headlines can create fast downside shock but short-lived follow-through,
- macro liquidity narratives may have slower but more persistent trend influence.

### 6.6 Why This Layer Is Essential

Without this layer, the system would only be a technical-indicator trader. Your requirement is stronger: the agent must combine **historical data, news, indicators, pre-analysis, and learned lessons** into one evolving decision framework.

---

## 7. Context Engineering for a Long-Horizon Trading Agent

### 7.1 Why Context Engineering Matters Here

This project spans three full years of market data and news. No LLM can reason safely by loading all raw candles and all raw articles into a single prompt. The system therefore needs strong context engineering.

### 7.2 Required Context Layers

- **Instructions**
  System rules, risk rules, allowed actions, reporting format.
- **Knowledge**
  Preprocessed market features, indicator tables, news summaries, impact-study outputs.
- **Working Memory**
  Current month state, current open trade state, regime summary, relevant recent news clusters.
- **Long-Term Memory**
  Monthly lessons, strategy revisions, known failure modes, high-performing setups, conditions to avoid.
- **Tool Outputs**
  Feature queries, sentiment results, paper-trade logs, scorecards, and backtest metrics.

### 7.3 Compression Strategy

The system should never feed raw historical bulk data directly to the LLM unless necessary. Instead it should compress:

- raw candles -> indicator tables and regime summaries,
- raw news -> event clusters and sentiment timelines,
- trade logs -> monthly performance summaries,
- lessons learned -> concise strategy revision memory.

This is what enables recurring self-improvement without context explosion.

---

## 8. Paper Trading Framework

### 8.1 Required Trading Logic

The paper-trading agent must operate exactly as requested:

- use **`4h` and `1d`** to identify overall market trend, BTC context, and macro situation,
- use **`15m` and `1h`** to determine entries and exits.

### 8.2 Decision Flow per Trade

For every potential trade, the agent should follow this sequence:

1. determine the `1d` market regime,
2. confirm or reject directional bias using `4h`,
3. inspect recent relevant news and active impact windows,
4. evaluate whether conditions favor trend-following, breakout, pullback, or mean-reversion,
5. seek entry confirmation on `1h`,
6. fine-tune trigger on `15m`,
7. define stop, take-profit, invalidation, and trade duration expectation,
8. simulate execution with realistic timing constraints,
9. manage trade exits using lower-timeframe conditions while respecting higher-timeframe invalidation.

### 8.3 Required Paper Trading Inputs

Each historical trade decision should be based on:

- market data,
- historical news available before the decision timestamp,
- pre-calculated indicators,
- pre-analyzed news sentiment,
- pre-analyzed market regime outputs,
- strategy memory from prior months only.

This avoids forward-looking leakage and preserves realistic historical simulation.

### 8.4 Trade Types

The strategy framework may support:

- long trend continuation,
- long pullback in uptrend,
- short trend continuation in downtrend,
- short rally fade in downtrend,
- volatility breakout,
- no-trade state when signals conflict.

No-trade discipline should be treated as a valid decision, not a failure.

---

## 9. Monthly Self-Learning and Self-Improvement Loop

### 9.1 Monthly Walk-Forward Design

The most important upgrade in this document is the **monthly recurring self-improvement loop**.

The system should process history month by month:

1. start with an initial strategy hypothesis,
2. paper trade one month of historical data,
3. measure performance and diagnose errors,
4. extract lessons,
5. update the strategy,
6. carry the updated strategy into the next month,
7. continue until the end of `2025-12`.

This creates a realistic walk-forward learning framework rather than a one-time optimization.

### 9.2 What the Agent Should Learn Each Month

The Reflection and Learning Agent should summarize:

- which setups worked,
- which setups failed,
- in which market regimes the strategy had edge,
- how news changed signal quality,
- whether entries were too early or too late,
- whether exits captured enough of the move,
- whether risk was too aggressive or too conservative,
- whether certain indicators added value or noise,
- whether impact duration estimates were accurate.

### 9.3 Monthly Improvement Actions

After extracting lessons, the system should update one or more of:

- indicator thresholds,
- regime filters,
- news sentiment weighting,
- cooldown periods after major news,
- stop-loss and take-profit logic,
- position sizing rules,
- confirmation requirements across timeframes,
- trade avoidance rules in low-quality conditions,
- ranking weights for setup selection.

### 9.4 Guardrails Against Overfitting

Self-improvement must not become blind curve-fitting. The monthly learning loop should therefore favor:

- small and explainable rule changes,
- limited parameter drift,
- documented rationale for every revision,
- validation across multiple prior months or regimes where possible,
- explicit rejection of improvements that help one month but damage robustness.

The goal is not to memorize history but to discover durable behavior.

---

## 10. Strategy Formation and Final Strategy Markdown

### 10.1 Purpose

After the month-by-month learning loop completes, the agent should consolidate its best and most robust logic into a **final trading strategy document** in markdown.

### 10.2 What the Final Strategy Document Should Contain

The final strategy markdown should include:

- strategy name,
- strategy objective,
- traded market and timeframe stack,
- market regime definitions,
- news-handling policy,
- required indicator set,
- long-entry rules,
- short-entry rules,
- no-trade rules,
- stop-loss rules,
- take-profit rules,
- position sizing rules,
- trade management rules,
- invalidation conditions,
- monthly review process,
- known weaknesses,
- best-performing environments,
- worst-performing environments.

### 10.3 Why a Written Strategy Matters

If the final strategy cannot be clearly written, it is not mature enough. A written strategy is the bridge between experimental agent behavior and auditable trading logic.

---

## 11. Final Backtest Requirement

### 11.1 Backtest Objective

Once the final strategy is drafted, the agent should run a full historical backtest across the prepared dataset using:

- historical trading data,
- historical news,
- pre-calculated indicators,
- pre-analyzed sentiment outputs,
- pre-analyzed impact-duration results,
- the final frozen strategy rules.

### 11.2 Backtest Principles

The final backtest should:

- use only time-appropriate information,
- avoid data leakage,
- respect the timeframe hierarchy,
- include realistic trading assumptions,
- keep the final strategy fixed during the final evaluation run.

### 11.3 Required Backtest Metrics

The final report should be as detailed as possible and include at minimum:

- total return,
- annualized return,
- Sharpe ratio,
- Sortino ratio,
- maximum drawdown,
- Calmar ratio,
- profit factor,
- win rate,
- average win,
- average loss,
- expectancy,
- average holding time,
- number of trades,
- monthly performance breakdown,
- regime-wise performance,
- long vs short performance,
- news-event trade performance,
- best trades,
- worst trades,
- streak analysis,
- risk-adjusted observations,
- failure modes and caveats.

### 11.4 Required Backtest Narrative

The report should not only show metrics. It should explain:

- why the strategy worked when it worked,
- where the edge came from,
- how the higher-timeframe filters improved lower-timeframe entries,
- how news intelligence changed outcomes,
- how the monthly self-improvement loop changed the final system,
- where the strategy remains fragile.

---

## 12. Detailed End-to-End Workflow

This is the full operating sequence the upgraded agent should follow.

### Phase 1: Data Acquisition

1. Download Binance BTCUSDT OHLCV data for `15m`, `1h`, `4h`, and `1d` from `2023-01-01` to `2025-12-31`.
2. Validate completeness, continuity, and timestamps.
3. Download or ingest historical BTC, crypto, and Web3 news for the same date range.
4. Normalize news timestamps, topics, source quality, and entity tags.

### Phase 2: Preprocessing and Feature Engineering

1. Pre-calculate all required indicators for each timeframe.
2. Build regime and structural features.
3. Cluster related news and score sentiment.
4. Estimate lag and duration of price impact per news item or event cluster.
5. Create compressed monthly context summaries for agent use.

### Phase 3: Monthly Paper Trading

1. Initialize starting strategy.
2. For each month from `2023-01` through `2025-12`:
   - read only data available up to that month,
   - identify `1d` and `4h` market context,
   - use `1h` and `15m` for entries and exits,
   - integrate relevant active news effects,
   - paper trade the month,
   - produce trade logs and monthly report.

### Phase 4: Monthly Reflection and Improvement

1. Evaluate monthly results.
2. Summarize lessons, edge sources, and recurring mistakes.
3. Update strategy rules conservatively.
4. Save updated strategy memory.
5. Roll forward to the next month.

### Phase 5: Final Strategy Consolidation

1. Compare all monthly revisions.
2. Keep robust rules and discard unstable ones.
3. Draft the final strategy markdown.

### Phase 6: Final Frozen Backtest

1. Lock the final strategy.
2. Run the full historical backtest.
3. Generate a detailed backtest report.

---

## 13. Recommended Evaluation Dimensions

The agent should score itself across multiple dimensions, not only profitability.

- **Return Quality**
  Net return, risk-adjusted return, consistency.
- **Risk Quality**
  drawdown control, tail-loss behavior, regime resilience.
- **Execution Quality**
  entry efficiency, exit efficiency, premature exit rate, late entry rate.
- **Signal Quality**
  alignment between higher and lower timeframes, false-positive rate, no-trade discipline.
- **News Intelligence Quality**
  sentiment accuracy, impact-window calibration, event relevance filtering.
- **Learning Quality**
  whether monthly changes improved robustness rather than just local fit.
- **Explainability**
  whether the system can clearly explain why each trade was taken or avoided.

These evaluation dimensions are essential for meaningful self-improvement.

---

## 14. Trustworthiness, Safety, and Research Integrity

Even though this is a historical paper-trading system, it still needs strong guardrails.

- **No data leakage**
  The agent must not access future candles, future news, or future monthly lessons.
- **Reproducibility**
  Every run should be traceable to the same data snapshot and rule version.
- **Explainability**
  Each strategy revision should have a written rationale.
- **Human auditability**
  A human reviewer should be able to inspect why the agent changed its rules.
- **Separation of roles**
  Data ingestion, feature computation, trading, and reflection should be separated to reduce hidden errors.

This is the quant-trading version of trustworthy AI-agent design.

---

## 15. Deliverables Required from the Upgraded Agent System

To fully satisfy the project requirements, the completed system should ultimately produce the following artifacts:

1. **Historical BTC market dataset** covering `2023-01-01` to `2025-12-31` for `15m`, `1h`, `4h`, and `1d`.
2. **Pre-calculated indicator dataset** for all required timeframes.
3. **Historical crypto/Web3/BTC news dataset** for the same period.
4. **News sentiment and impact analysis outputs** including estimated effect timing and duration.
5. **Monthly paper-trading logs** with timestamped rationale.
6. **Monthly performance reports** with lessons learned.
7. **Monthly strategy revision history** showing how the agent self-improved over time.
8. **Final strategy markdown** describing the learned trading strategy.
9. **Final backtest report** with detailed metrics, narrative analysis, strengths, weaknesses, and risk caveats.

---

## 16. Exact Mapping to the 9 Required Capabilities

### Requirement 1

The agent is designed to **recurringly self-learn and self-improve** using historical market data, historical news, monthly reflection, and long-term strategy memory.

### Requirement 2

The Market Data Agent explicitly downloads **Binance free API** BTC historical price and volume data from `2023-01-01` to `2025-12-31` for `15m`, `1h`, `4h`, and `1d`.

### Requirement 3

The Feature Engineering Agent pre-calculates the required indicators including **MA/EMA/SMA, MACD, RSI, Bollinger Bands, Ichimoku Cloud, VWAP, Volume Profile**, and related derived features.

### Requirement 4

The News Intelligence Agent downloads historical BTC/crypto/Web3 news for `2023-01-01` to `2025-12-31`, analyzes sentiment, and estimates **how** and **how long** the news affects BTC price.

### Requirement 5

The Paper Trading Agent uses historical prices, historical news, indicators, and analysis outputs to perform **historical paper trading** and generate detailed trading reports.

### Requirement 6

The trading logic explicitly uses **`4h` and `1d`** for market trend and situation analysis, and **`15m` and `1h`** for entries and exits.

### Requirement 7

The monthly walk-forward loop ensures the agent keeps trading historical data month by month, summarizes performance, learns lessons, self-improves, and applies the updated strategy to the next month.

### Requirement 8

After the learning cycle completes, the Strategy Authoring Agent drafts the **final trading strategy as a markdown file**.

### Requirement 9

The final frozen strategy is then used for a full **backtest** on historical trading data, historical news, preprocessed indicators, and pre-analysis outputs, and the result is written as a detailed backtest report.

---

## 17. Final Conclusion

The upgraded design is no longer just “an AI agent with indicators.” It is a **self-improving quant research agent system** built around a disciplined learning loop:

**historical data + historical news + multi-timeframe indicators + paper trading + monthly reflection + strategy updates + final strategy drafting + frozen backtest**

That closed loop is the key architectural upgrade. It transforms the system from static rule testing into a recurring learning framework capable of producing a refined BTC trading strategy and a detailed evidence trail for how that strategy was formed.
