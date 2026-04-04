# Monthly Paper Trading Report: 2025-04

- Total Return: -0.36%
- Sharpe: -2.9969
- Max Drawdown: 0.39%
- Win Rate: 0.00%
- Profit Factor: 0.0000
- Expectancy: -0.36%
- Trade Count: 1
- Reliability Flags: low_sample_size, one_sided_exposure, extreme_win_rate_small_sample, negative_sharpe, profit_factor_below_one, negative_total_return

## Lesson Learned

Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.

## Strategy Snapshot

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
