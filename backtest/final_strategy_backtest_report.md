# Final Frozen Backtest Report

- Generated At: 2026-04-04T02:08:14.335207+00:00
- Total Return: -4.20%
- Annualized Return: -1.42%
- Sharpe: -0.8295
- Sortino: -0.1226
- Max Drawdown: 4.80%
- Calmar: -0.2958
- Win Rate: 25.00%
- Profit Factor: 0.6445
- Expectancy: -0.07%
- Average Win: 0.51%
- Average Loss: -0.27%
- Average Holding Bars: 16.4167
- Trade Count: 60
- Reliability Flags: negative_sharpe, profit_factor_below_one, negative_total_return

## Strategy Parameters

- `bb_period`: 20
- `bb_std`: 2
- `breakeven_r_multiple`: 1.0
- `breakout_15m_window`: 32
- `breakout_1h_window`: 72
- `breakout_volume_surge`: 1.2
- `conflict_penalty`: 0.45
- `cooldown_bars_after_news`: 8
- `ema_long`: 26
- `ema_short`: 12
- `entry_timeframe`: 15m
- `ichimoku_base_period`: 26
- `ichimoku_conversion_period`: 9
- `ichimoku_span_b_period`: 52
- `long_pullback_rsi_15m`: 44
- `long_pullback_rsi_1h`: 46
- `long_score_threshold`: 0.82
- `ma_long`: 50
- `ma_short`: 10
- `macd_signal`: 9.0
- `max_hold_bars`: 192
- `max_position`: 0.55
- `news_impact_cap`: 0.6
- `news_weight`: 0.15
- `rsi_buy`: 40.0
- `rsi_fast_period`: 7
- `rsi_period`: 14
- `rsi_sell`: 65.0
- `score_hysteresis`: 0.08
- `short_breakdown_volume_surge`: 1.6
- `short_rally_rsi_1h`: 58
- `short_score_threshold`: 0.95
- `sma_long`: 50
- `sma_short`: 10
- `stop_atr_multiple`: 2.6
- `strategy_name`: btc_multi_tf_news_regime_v1
- `take_profit_atr_multiple`: 5.0
- `trade_cooldown_bars`: 32
- `trend_filter_strength`: 0.62
- `volume_profile_window`: 96
- `vwap_window`: 20
- `weight_resonance`: 1.08

## Monthly Breakdown

- `2023-01` return=-0.32%, max_dd=0.60%, trades=5
- `2023-02` return=0.00%, max_dd=0.57%, trades=0
- `2023-03` return=-0.60%, max_dd=1.26%, trades=4
- `2023-04` return=0.24%, max_dd=1.71%, trades=3
- `2023-05` return=0.00%, max_dd=0.93%, trades=0
- `2023-06` return=0.93%, max_dd=0.96%, trades=2
- `2023-07` return=-0.45%, max_dd=0.62%, trades=2
- `2023-08` return=0.00%, max_dd=0.62%, trades=0
- `2023-09` return=0.00%, max_dd=0.62%, trades=0
- `2023-10` return=-0.85%, max_dd=1.55%, trades=3
- `2023-11` return=-0.07%, max_dd=2.18%, trades=6
- `2023-12` return=-0.83%, max_dd=2.35%, trades=2
- `2024-01` return=-0.05%, max_dd=2.40%, trades=1
- `2024-02` return=1.74%, max_dd=2.56%, trades=3
- `2024-03` return=-0.16%, max_dd=0.97%, trades=1
- `2024-04` return=0.00%, max_dd=0.86%, trades=0
- `2024-05` return=0.00%, max_dd=0.86%, trades=0
- `2024-06` return=-0.23%, max_dd=1.08%, trades=1
- `2024-07` return=-1.10%, max_dd=2.17%, trades=2
- `2024-08` return=0.00%, max_dd=2.17%, trades=0
- `2024-09` return=0.51%, max_dd=2.18%, trades=1
- `2024-10` return=0.74%, max_dd=1.93%, trades=2
- `2024-11` return=-0.76%, max_dd=1.90%, trades=3
- `2024-12` return=-0.41%, max_dd=2.10%, trades=1
- `2025-01` return=1.38%, max_dd=2.12%, trades=2
- `2025-02` return=0.00%, max_dd=0.75%, trades=0
- `2025-03` return=-0.36%, max_dd=1.10%, trades=1
- `2025-04` return=-0.36%, max_dd=1.45%, trades=1
- `2025-05` return=-1.16%, max_dd=2.65%, trades=3
- `2025-06` return=-0.29%, max_dd=2.88%, trades=1
- `2025-07` return=0.07%, max_dd=3.10%, trades=3
- `2025-08` return=-0.70%, max_dd=3.49%, trades=2
- `2025-09` return=-0.25%, max_dd=3.74%, trades=1
- `2025-10` return=-0.89%, max_dd=4.80%, trades=4
- `2025-11` return=0.00%, max_dd=4.59%, trades=0
- `2025-12` return=0.00%, max_dd=4.59%, trades=0

## Regime Breakdown

- `long` trades=57, win_rate=22.81%, return=-4.63%
- `short` trades=3, win_rate=66.67%, return=0.38%

## Trade Details

### Trade 1 (Long)

- Entry Time: 2023-01-02T04:00:00+00:00
- Exit Time: 2023-01-02T14:45:00+00:00
- Entry Price: 16,661.94
- Exit Price: 16,702.56
- Size: 0.5500
- Return: 0.01%
- PnL: 0.00%
- Bars Held: 44
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9093/0.5330, RSI=74.1920, MACD=4.7985
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 2 (Long)

- Entry Time: 2023-01-04T04:00:00+00:00
- Exit Time: 2023-01-04T12:45:00+00:00
- Entry Price: 16,862.02
- Exit Price: 16,826.67
- Size: 0.5500
- Return: -0.19%
- PnL: -0.10%
- Bars Held: 36
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9417/0.3430, RSI=77.5802, MACD=40.5017
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 3 (Long)

- Entry Time: 2023-01-09T00:00:00+00:00
- Exit Time: 2023-01-09T14:00:00+00:00
- Entry Price: 17,127.83
- Exit Price: 17,226.90
- Size: 0.5500
- Return: 0.22%
- PnL: 0.12%
- Bars Held: 57
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9741/0.2820, RSI=84.6523, MACD=30.4649
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 4 (Long)

- Entry Time: 2023-01-12T01:00:00+00:00
- Exit Time: 2023-01-12T06:00:00+00:00
- Entry Price: 18,264.10
- Exit Price: 18,087.54
- Size: 0.3513
- Return: -0.36%
- PnL: -0.13%
- Bars Held: 21
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9741/0.2460, RSI=90.8171, MACD=149.9084
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.3513.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 5 (Long)

- Entry Time: 2023-01-20T20:00:00+00:00
- Exit Time: 2023-01-20T20:15:00+00:00
- Entry Price: 21,496.67
- Exit Price: 21,850.88
- Size: 0.4220
- Return: 0.00%
- PnL: 0.00%
- Bars Held: 2
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9417/0.2710, RSI=78.6590, MACD=88.4915
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.4220.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 6 (Short)

- Entry Time: 2023-03-09T17:00:00+00:00
- Exit Time: 2023-03-09T18:45:00+00:00
- Entry Price: 21,478.85
- Exit Price: 21,074.49
- Size: 0.1375
- Return: 0.23%
- PnL: 0.03%
- Bars Held: 8
- Higher TF Context: 4h=bearish, 1d=bearish
- Lower TF Trigger: Regime: bear, Entry workflow: 1h+15m, setup=short_breakdown, score(L/S)=0.0000/1.0000, RSI=26.1637, MACD=-26.5653
- News Filter: No active news events applied.
- Entry Rationale: Opened short position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed short_breakdown, and signal strength after news/regime adjustment was -0.1375.
- Exit Rationale: Exited because ATR take-profit was hit.

### Trade 7 (Long)

- Entry Time: 2023-03-21T13:00:00+00:00
- Exit Time: 2023-03-21T14:30:00+00:00
- Entry Price: 28,188.92
- Exit Price: 27,855.77
- Size: 0.3102
- Return: -0.54%
- PnL: -0.17%
- Bars Held: 7
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=1.0000/0.0360, RSI=67.2667, MACD=96.6261
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.3102.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 8 (Long)

- Entry Time: 2023-03-26T12:00:00+00:00
- Exit Time: 2023-03-26T14:45:00+00:00
- Entry Price: 27,865.95
- Exit Price: 27,806.44
- Size: 0.5500
- Return: -0.01%
- PnL: -0.00%
- Bars Held: 12
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9644/0.0970, RSI=72.3764, MACD=57.0132
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 9 (Long)

- Entry Time: 2023-03-28T20:00:00+00:00
- Exit Time: 2023-03-29T00:00:00+00:00
- Entry Price: 27,425.99
- Exit Price: 27,261.07
- Size: 0.3894
- Return: -0.28%
- PnL: -0.11%
- Bars Held: 17
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9644/0.1220, RSI=73.5391, MACD=98.7586
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.3894.
- Exit Rationale: Exited because 4h/1d regime invalidated the trade.

### Trade 10 (Long)

- Entry Time: 2023-04-09T22:00:00+00:00
- Exit Time: 2023-04-10T05:30:00+00:00
- Entry Price: 28,437.19
- Exit Price: 28,233.27
- Size: 0.5500
- Return: -0.53%
- PnL: -0.29%
- Bars Held: 31
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9513/0.1400, RSI=82.0839, MACD=103.3178
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 11 (Long)

- Entry Time: 2023-04-10T17:00:00+00:00
- Exit Time: 2023-04-10T23:15:00+00:00
- Entry Price: 29,036.85
- Exit Price: 29,747.38
- Size: 0.4282
- Return: 1.00%
- PnL: 0.43%
- Bars Held: 26
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9833/0.0950, RSI=83.9403, MACD=96.1031
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.4282.
- Exit Rationale: Exited because ATR take-profit was hit.

### Trade 12 (Long)

- Entry Time: 2023-04-16T19:00:00+00:00
- Exit Time: 2023-04-16T20:00:00+00:00
- Entry Price: 30,478.43
- Exit Price: 30,309.06
- Size: 0.5500
- Return: -0.23%
- PnL: -0.13%
- Bars Held: 5
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9644/0.0970, RSI=74.5641, MACD=23.5449
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 13 (Long)

- Entry Time: 2023-06-23T15:00:00+00:00
- Exit Time: 2023-06-23T15:30:00+00:00
- Entry Price: 30,282.04
- Exit Price: 30,650.00
- Size: 0.5319
- Return: 0.99%
- PnL: 0.53%
- Bars Held: 3
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9741/0.1060, RSI=64.9105, MACD=36.6155
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5319.
- Exit Rationale: Exited because ATR take-profit was hit.

### Trade 14 (Long)

- Entry Time: 2023-06-27T12:00:00+00:00
- Exit Time: 2023-06-27T14:15:00+00:00
- Entry Price: 30,729.10
- Exit Price: 30,752.01
- Size: 0.5500
- Return: -0.07%
- PnL: -0.04%
- Bars Held: 10
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9417/0.1310, RSI=67.7324, MACD=84.1272
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 15 (Long)

- Entry Time: 2023-07-03T19:00:00+00:00
- Exit Time: 2023-07-03T21:30:00+00:00
- Entry Price: 31,199.14
- Exit Price: 31,025.81
- Size: 0.4951
- Return: -0.38%
- PnL: -0.19%
- Bars Held: 11
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=1.0000/0.0700, RSI=70.3901, MACD=101.0066
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.4951.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 16 (Long)

- Entry Time: 2023-07-10T20:00:00+00:00
- Exit Time: 2023-07-10T20:15:00+00:00
- Entry Price: 30,837.40
- Exit Price: 30,888.00
- Size: 0.5500
- Return: -0.07%
- PnL: -0.04%
- Bars Held: 2
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9513/0.1400, RSI=83.6097, MACD=112.0873
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 17 (Long)

- Entry Time: 2023-10-18T05:00:00+00:00
- Exit Time: 2023-10-18T08:00:00+00:00
- Entry Price: 28,730.72
- Exit Price: 28,494.55
- Size: 0.5500
- Return: -0.47%
- PnL: -0.26%
- Bars Held: 13
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9741/0.1060, RSI=74.9042, MACD=39.1296
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 18 (Long)

- Entry Time: 2023-10-23T04:00:00+00:00
- Exit Time: 2023-10-23T08:00:00+00:00
- Entry Price: 30,736.69
- Exit Price: 30,518.02
- Size: 0.3550
- Return: -0.32%
- PnL: -0.12%
- Bars Held: 17
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=1.0000/0.0700, RSI=75.4933, MACD=157.1179
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.3550.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 19 (Long)

- Entry Time: 2023-10-29T13:00:00+00:00
- Exit Time: 2023-10-29T16:00:00+00:00
- Entry Price: 34,419.99
- Exit Price: 34,449.99
- Size: 0.5500
- Return: -0.07%
- PnL: -0.04%
- Bars Held: 13
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9417/0.1310, RSI=76.0167, MACD=65.5227
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 20 (Long)

- Entry Time: 2023-11-01T13:00:00+00:00
- Exit Time: 2023-11-01T13:30:00+00:00
- Entry Price: 34,790.61
- Exit Price: 34,736.26
- Size: 0.5500
- Return: -0.07%
- PnL: -0.04%
- Bars Held: 3
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9417/0.1310, RSI=76.0718, MACD=20.3786
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 21 (Long)

- Entry Time: 2023-11-05T04:00:00+00:00
- Exit Time: 2023-11-05T06:45:00+00:00
- Entry Price: 35,277.01
- Exit Price: 35,059.74
- Size: 0.5500
- Return: -0.36%
- PnL: -0.20%
- Bars Held: 12
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=1.0000/0.0250, RSI=71.9727, MACD=69.3737
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 22 (Long)

- Entry Time: 2023-11-08T19:00:00+00:00
- Exit Time: 2023-11-08T21:15:00+00:00
- Entry Price: 35,589.57
- Exit Price: 35,598.00
- Size: 0.5500
- Return: -0.07%
- PnL: -0.04%
- Bars Held: 10
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9833/0.0950, RSI=66.9206, MACD=38.5956
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 23 (Long)

- Entry Time: 2023-11-09T14:00:00+00:00
- Exit Time: 2023-11-09T14:15:00+00:00
- Entry Price: 37,165.98
- Exit Price: 37,597.30
- Size: 0.5500
- Return: -0.02%
- PnL: -0.01%
- Bars Held: 2
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=1.0000/0.0700, RSI=74.8396, MACD=110.6873
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 24 (Long)

- Entry Time: 2023-11-19T20:00:00+00:00
- Exit Time: 2023-11-19T23:15:00+00:00
- Entry Price: 36,948.97
- Exit Price: 37,384.26
- Size: 0.5500
- Return: 0.12%
- PnL: 0.07%
- Bars Held: 14
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9741/0.1060, RSI=77.3809, MACD=93.5465
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 25 (Long)

- Entry Time: 2023-11-24T08:00:00+00:00
- Exit Time: 2023-11-24T10:30:00+00:00
- Entry Price: 37,522.99
- Exit Price: 37,713.29
- Size: 0.5500
- Return: 0.32%
- PnL: 0.18%
- Bars Held: 11
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=1.0000/0.0700, RSI=65.6999, MACD=26.9249
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR take-profit was hit.

### Trade 26 (Long)

- Entry Time: 2023-12-13T22:00:00+00:00
- Exit Time: 2023-12-14T00:45:00+00:00
- Entry Price: 42,988.49
- Exit Price: 42,666.26
- Size: 0.3796
- Return: -0.28%
- PnL: -0.10%
- Bars Held: 12
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9417/0.1560, RSI=73.5313, MACD=308.9113
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.3796.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 27 (Long)

- Entry Time: 2023-12-19T02:00:00+00:00
- Exit Time: 2023-12-19T06:30:00+00:00
- Entry Price: 43,314.99
- Exit Price: 42,897.46
- Size: 0.4432
- Return: -0.55%
- PnL: -0.25%
- Bars Held: 19
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9417/0.1310, RSI=80.7299, MACD=290.3024
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.4432.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 28 (Long)

- Entry Time: 2024-01-08T12:00:00+00:00
- Exit Time: 2024-01-08T12:15:00+00:00
- Entry Price: 44,635.99
- Exit Price: 45,141.09
- Size: 0.5381
- Return: -0.05%
- PnL: -0.02%
- Bars Held: 2
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9741/0.1060, RSI=75.5812, MACD=139.9837
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5381.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 29 (Long)

- Entry Time: 2024-02-09T09:00:00+00:00
- Exit Time: 2024-02-09T12:15:00+00:00
- Entry Price: 46,639.99
- Exit Price: 47,420.89
- Size: 0.5500
- Return: 0.78%
- PnL: 0.43%
- Bars Held: 14
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=1.0000/0.0000, RSI=73.1570, MACD=191.5358
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR take-profit was hit.

### Trade 30 (Long)

- Entry Time: 2024-02-14T09:00:00+00:00
- Exit Time: 2024-02-14T09:15:00+00:00
- Entry Price: 50,758.00
- Exit Price: 50,964.01
- Size: 0.5394
- Return: -0.06%
- PnL: -0.03%
- Bars Held: 2
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=1.0000/0.0360, RSI=83.9883, MACD=171.2018
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5394.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 31 (Long)

- Entry Time: 2024-02-28T15:00:00+00:00
- Exit Time: 2024-02-28T17:15:00+00:00
- Entry Price: 60,971.11
- Exit Price: 63,690.12
- Size: 0.2331
- Return: 1.02%
- PnL: 0.24%
- Bars Held: 10
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9741/0.1060, RSI=78.6277, MACD=585.8096
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.2331.
- Exit Rationale: Exited because ATR take-profit was hit.

### Trade 32 (Long)

- Entry Time: 2024-03-03T17:00:00+00:00
- Exit Time: 2024-03-03T23:00:00+00:00
- Entry Price: 62,845.15
- Exit Price: 62,827.43
- Size: 0.5500
- Return: -0.16%
- PnL: -0.09%
- Bars Held: 25
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9741/0.1060, RSI=69.9135, MACD=158.4011
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 33 (Long)

- Entry Time: 2024-06-04T16:00:00+00:00
- Exit Time: 2024-06-04T19:45:00+00:00
- Entry Price: 70,445.00
- Exit Price: 70,336.01
- Size: 0.3980
- Return: -0.23%
- PnL: -0.09%
- Bars Held: 16
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9513/0.1400, RSI=78.9368, MACD=303.3934
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.3980.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 34 (Long)

- Entry Time: 2024-07-21T17:00:00+00:00
- Exit Time: 2024-07-21T18:00:00+00:00
- Entry Price: 67,554.00
- Exit Price: 66,648.01
- Size: 0.5176
- Return: -0.56%
- PnL: -0.29%
- Bars Held: 5
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9741/0.1060, RSI=66.6164, MACD=94.1659
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5176.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 35 (Long)

- Entry Time: 2024-07-22T20:00:00+00:00
- Exit Time: 2024-07-22T22:15:00+00:00
- Entry Price: 68,232.07
- Exit Price: 67,625.33
- Size: 0.5500
- Return: -0.54%
- PnL: -0.29%
- Bars Held: 10
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9741/0.1060, RSI=73.7414, MACD=166.7589
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 36 (Short)

- Entry Time: 2024-09-06T15:00:00+00:00
- Exit Time: 2024-09-06T21:00:00+00:00
- Entry Price: 54,733.24
- Exit Price: 52,887.99
- Size: 0.1375
- Return: 0.51%
- PnL: 0.07%
- Bars Held: 25
- Higher TF Context: 4h=bearish, 1d=bearish
- Lower TF Trigger: Regime: bear, Entry workflow: 1h+15m, setup=short_breakdown, score(L/S)=0.0000/1.0000, RSI=27.6700, MACD=-274.1681
- News Filter: No active news events applied.
- Entry Rationale: Opened short position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed short_breakdown, and signal strength after news/regime adjustment was -0.1375.
- Exit Rationale: Exited because ATR take-profit was hit.

### Trade 37 (Long)

- Entry Time: 2024-10-20T21:00:00+00:00
- Exit Time: 2024-10-20T22:15:00+00:00
- Entry Price: 68,781.98
- Exit Price: 68,721.15
- Size: 0.5500
- Return: -0.07%
- PnL: -0.04%
- Bars Held: 6
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=1.0000/0.0360, RSI=66.9256, MACD=45.2899
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 38 (Long)

- Entry Time: 2024-10-29T14:00:00+00:00
- Exit Time: 2024-10-29T17:15:00+00:00
- Entry Price: 71,804.00
- Exit Price: 72,745.51
- Size: 0.5500
- Return: 0.81%
- PnL: 0.45%
- Bars Held: 14
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=1.0000/0.0610, RSI=67.0860, MACD=114.8461
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR take-profit was hit.

### Trade 39 (Long)

- Entry Time: 2024-11-15T21:00:00+00:00
- Exit Time: 2024-11-16T11:30:00+00:00
- Entry Price: 91,447.98
- Exit Price: 91,220.00
- Size: 0.3188
- Return: -0.15%
- PnL: -0.05%
- Bars Held: 59
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=1.0000/0.0360, RSI=70.2226, MACD=485.2352
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.3188.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 40 (Long)

- Entry Time: 2024-11-18T16:00:00+00:00
- Exit Time: 2024-11-18T18:45:00+00:00
- Entry Price: 92,309.51
- Exit Price: 90,450.09
- Size: 0.2623
- Return: -0.53%
- PnL: -0.14%
- Bars Held: 12
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=1.0000/0.0360, RSI=65.9325, MACD=77.9964
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.2623.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 41 (Long)

- Entry Time: 2024-11-21T05:00:00+00:00
- Exit Time: 2024-11-21T13:00:00+00:00
- Entry Price: 97,554.22
- Exit Price: 97,317.93
- Size: 0.3476
- Return: -0.08%
- PnL: -0.03%
- Bars Held: 33
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=1.0000/0.0000, RSI=76.0456, MACD=673.5954
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.3476.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 42 (Long)

- Entry Time: 2024-12-09T00:00:00+00:00
- Exit Time: 2024-12-09T00:30:00+00:00
- Entry Price: 101,109.59
- Exit Price: 100,044.37
- Size: 0.5500
- Return: -0.41%
- PnL: -0.23%
- Bars Held: 3
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9741/0.1060, RSI=76.2161, MACD=212.2905
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 43 (Long)

- Entry Time: 2025-01-06T15:00:00+00:00
- Exit Time: 2025-01-07T01:30:00+00:00
- Entry Price: 100,737.03
- Exit Price: 101,685.27
- Size: 0.4273
- Return: 0.39%
- PnL: 0.17%
- Bars Held: 43
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9741/0.1060, RSI=71.5198, MACD=162.3677
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.4273.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 44 (Long)

- Entry Time: 2025-01-17T15:00:00+00:00
- Exit Time: 2025-01-17T19:30:00+00:00
- Entry Price: 103,378.95
- Exit Price: 105,409.29
- Size: 0.5384
- Return: 0.99%
- PnL: 0.53%
- Bars Held: 19
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=1.0000/0.0360, RSI=69.7482, MACD=286.3548
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5384.
- Exit Rationale: Exited because ATR take-profit was hit.

### Trade 45 (Short)

- Entry Time: 2025-03-10T19:00:00+00:00
- Exit Time: 2025-03-10T21:00:00+00:00
- Entry Price: 77,655.26
- Exit Price: 79,304.98
- Size: 0.1375
- Return: -0.36%
- PnL: -0.05%
- Bars Held: 9
- Higher TF Context: 4h=bearish, 1d=bearish
- Lower TF Trigger: Regime: bear, Entry workflow: 1h+15m, setup=short_breakdown, score(L/S)=0.0416/0.9640, RSI=26.2346, MACD=-906.4333
- News Filter: No active news events applied.
- Entry Rationale: Opened short position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed short_breakdown, and signal strength after news/regime adjustment was -0.1375.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 46 (Long)

- Entry Time: 2025-04-25T15:00:00+00:00
- Exit Time: 2025-04-25T22:00:00+00:00
- Entry Price: 95,368.00
- Exit Price: 94,608.69
- Size: 0.3424
- Return: -0.36%
- PnL: -0.12%
- Bars Held: 29
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9741/0.1060, RSI=65.2868, MACD=357.0441
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.3424.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 47 (Long)

- Entry Time: 2025-05-18T17:00:00+00:00
- Exit Time: 2025-05-18T18:00:00+00:00
- Entry Price: 105,822.22
- Exit Price: 104,836.67
- Size: 0.5385
- Return: -0.54%
- PnL: -0.29%
- Bars Held: 5
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9513/0.1400, RSI=81.6872, MACD=468.9083
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5385.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 48 (Long)

- Entry Time: 2025-05-21T15:00:00+00:00
- Exit Time: 2025-05-21T17:30:00+00:00
- Entry Price: 108,999.99
- Exit Price: 106,975.90
- Size: 0.4893
- Return: -0.56%
- PnL: -0.27%
- Bars Held: 11
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=1.0000/0.0700, RSI=79.6409, MACD=325.4774
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.4893.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 49 (Long)

- Entry Time: 2025-05-26T01:00:00+00:00
- Exit Time: 2025-05-26T16:30:00+00:00
- Entry Price: 109,540.46
- Exit Price: 109,331.90
- Size: 0.4483
- Return: -0.07%
- PnL: -0.03%
- Bars Held: 63
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9417/0.1310, RSI=71.6938, MACD=455.9093
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.4483.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 50 (Long)

- Entry Time: 2025-06-29T12:00:00+00:00
- Exit Time: 2025-06-29T13:45:00+00:00
- Entry Price: 108,477.11
- Exit Price: 108,152.57
- Size: 0.5500
- Return: -0.29%
- PnL: -0.16%
- Bars Held: 8
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9644/0.0970, RSI=81.5892, MACD=242.1358
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 51 (Long)

- Entry Time: 2025-07-02T16:00:00+00:00
- Exit Time: 2025-07-02T16:15:00+00:00
- Entry Price: 108,737.74
- Exit Price: 109,420.02
- Size: 0.5500
- Return: -0.07%
- PnL: -0.04%
- Bars Held: 2
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9189/0.1650, RSI=74.8890, MACD=313.3933
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 52 (Long)

- Entry Time: 2025-07-13T13:00:00+00:00
- Exit Time: 2025-07-13T14:30:00+00:00
- Entry Price: 118,432.11
- Exit Price: 119,150.00
- Size: 0.5500
- Return: 0.31%
- PnL: 0.17%
- Bars Held: 7
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=1.0000/0.0360, RSI=71.4465, MACD=114.3175
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR take-profit was hit.

### Trade 53 (Long)

- Entry Time: 2025-07-22T19:00:00+00:00
- Exit Time: 2025-07-23T01:00:00+00:00
- Entry Price: 119,733.69
- Exit Price: 119,546.56
- Size: 0.5500
- Return: -0.16%
- PnL: -0.09%
- Bars Held: 25
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=1.0000/0.0700, RSI=63.3047, MACD=180.8371
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 54 (Long)

- Entry Time: 2025-08-12T21:00:00+00:00
- Exit Time: 2025-08-13T01:15:00+00:00
- Entry Price: 120,192.10
- Exit Price: 119,356.00
- Size: 0.5500
- Return: -0.32%
- PnL: -0.18%
- Bars Held: 18
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9741/0.1060, RSI=64.4126, MACD=191.1409
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 55 (Long)

- Entry Time: 2025-08-13T14:00:00+00:00
- Exit Time: 2025-08-13T15:00:00+00:00
- Entry Price: 121,767.97
- Exit Price: 120,890.73
- Size: 0.5500
- Return: -0.38%
- PnL: -0.21%
- Bars Held: 5
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=1.0000/0.0610, RSI=73.7076, MACD=285.2722
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 56 (Long)

- Entry Time: 2025-09-17T06:00:00+00:00
- Exit Time: 2025-09-17T08:45:00+00:00
- Entry Price: 117,108.30
- Exit Price: 116,759.20
- Size: 0.5500
- Return: -0.25%
- PnL: -0.14%
- Bars Held: 12
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=1.0000/0.0250, RSI=66.5956, MACD=20.1567
- News Filter: No active news events applied.
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 57 (Long)

- Entry Time: 2025-10-02T00:00:00+00:00
- Exit Time: 2025-10-02T00:15:00+00:00
- Entry Price: 118,594.99
- Exit Price: 119,153.80
- Size: 0.5500
- Return: -0.07%
- PnL: -0.04%
- Bars Held: 2
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9741/0.1060, RSI=73.0523, MACD=301.3530
- News Filter: 1 active news events, net bullish impact 0.1469
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 58 (Long)

- Entry Time: 2025-10-02T19:00:00+00:00
- Exit Time: 2025-10-03T01:30:00+00:00
- Entry Price: 120,506.61
- Exit Price: 120,119.23
- Size: 0.5500
- Return: -0.28%
- PnL: -0.15%
- Bars Held: 27
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9741/0.1060, RSI=66.7169, MACD=291.6989
- News Filter: 1 active news events, net bullish impact 0.1469
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 59 (Long)

- Entry Time: 2025-10-05T05:00:00+00:00
- Exit Time: 2025-10-05T09:30:00+00:00
- Entry Price: 125,172.81
- Exit Price: 124,013.40
- Size: 0.4800
- Return: -0.51%
- PnL: -0.24%
- Bars Held: 19
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9741/0.1060, RSI=81.5950, MACD=600.8272
- News Filter: 1 active news events, net bullish impact 0.1469
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.4800.
- Exit Rationale: Exited because ATR stop-loss was hit.

### Trade 60 (Long)

- Entry Time: 2025-10-06T14:00:00+00:00
- Exit Time: 2025-10-06T20:30:00+00:00
- Entry Price: 125,031.33
- Exit Price: 125,137.46
- Size: 0.5500
- Return: -0.04%
- PnL: -0.02%
- Bars Held: 27
- Higher TF Context: 4h=bullish, 1d=bullish
- Lower TF Trigger: Regime: bull, Entry workflow: 1h+15m, setup=long_breakout, score(L/S)=0.9741/0.1060, RSI=67.5702, MACD=262.2586
- News Filter: 1 active news events, net bullish impact 0.1469
- Entry Rationale: Opened long position because the 1d macro bias and 4h confirmation allowed the direction, the lower-timeframe workflow (1h+15m) confirmed long_breakout, and signal strength after news/regime adjustment was 0.5500.
- Exit Rationale: Exited because ATR stop-loss was hit.


## Narrative Analysis

The strategy uses a slow 1d macro bias with 4h confirmation, then confirms setups on 1h and executes on 15m. News impact windows modulate risk asymmetrically rather than acting as standalone triggers. Reliability caution: negative_sharpe, profit_factor_below_one, negative_total_return.
