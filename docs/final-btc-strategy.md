# Final BTC Strategy

- Asset: `BTCUSDT`
- Timeframes: `4h` + `1d` for trend context; `15m` + `1h` for entries/exits
- Data Range: `2023-01-01` to `2025-12-31`

## Rules

- `1d` defines the macro bias with a slow SMA-driven bull/bear filter and `4h` confirms direction.
- `1h` confirms the setup family and `15m` fine-tunes trigger timing for live entries and exits.
- BTC is treated as structurally long-biased unless the daily macro regime is decisively bearish.
- News sentiment is used as an asymmetric risk modifier and cooldown filter, not a standalone trigger.
- Risk controls use ATR-based stop logic, capped position sizing, and setup-aware target management.

## Final Parameters

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

## Monthly Learning Summary

- `2023-01` score=8.0938: Loss-making month with weak trade quality. Reduce countertrend short exposure, prefer bull-regime long breakout or continuation setups over shallow pullbacks, and trim news-driven signal amplification unless timestamp confidence is high.
- `2023-02` score=10.0000: No trades executed this month. Relax only the 15m trigger quality threshold or breakout volume filter modestly, keep the 1d macro filter intact, and keep shorts highly restrictive.
- `2023-03` score=-12.9565: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2023-04` score=40.9239: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2023-05` score=15.0000: No trades executed this month. Relax only the 15m trigger quality threshold or breakout volume filter modestly, keep the 1d macro filter intact, and keep shorts highly restrictive.
- `2023-06` score=74.4943: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2023-07` score=-35.4562: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2023-08` score=15.0000: No trades executed this month. Relax only the 15m trigger quality threshold or breakout volume filter modestly, keep the 1d macro filter intact, and keep shorts highly restrictive.
- `2023-09` score=15.0000: No trades executed this month. Relax only the 15m trigger quality threshold or breakout volume filter modestly, keep the 1d macro filter intact, and keep shorts highly restrictive.
- `2023-10` score=-53.1320: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2023-11` score=15.0910: Loss-making month with weak trade quality. Reduce countertrend short exposure, prefer bull-regime long breakout or continuation setups over shallow pullbacks, and trim news-driven signal amplification unless timestamp confidence is high.
- `2023-12` score=-51.0163: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2024-01` score=-48.2521: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2024-02` score=132.3041: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2024-03` score=4.6065: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2024-04` score=10.0000: No trades executed this month. Relax only the 15m trigger quality threshold or breakout volume filter modestly, keep the 1d macro filter intact, and keep shorts highly restrictive.
- `2024-05` score=15.0000: No trades executed this month. Relax only the 15m trigger quality threshold or breakout volume filter modestly, keep the 1d macro filter intact, and keep shorts highly restrictive.
- `2024-06` score=-4.4397: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2024-07` score=-66.0755: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2024-08` score=12.5000: No trades executed this month. Relax only the 15m trigger quality threshold or breakout volume filter modestly, keep the 1d macro filter intact, and keep shorts highly restrictive.
- `2024-09` score=105.8941: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2024-10` score=102.6120: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2024-11` score=-40.2498: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2024-12` score=-54.6459: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2025-01` score=98.2211: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2025-02` score=12.5000: No trades executed this month. Relax only the 15m trigger quality threshold or breakout volume filter modestly, keep the 1d macro filter intact, and keep shorts highly restrictive.
- `2025-03` score=-54.0865: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2025-04` score=-27.5242: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2025-05` score=-42.2611: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2025-06` score=-50.5239: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2025-07` score=32.8074: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2025-08` score=-59.4609: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2025-09` score=-52.9930: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2025-10` score=-40.2178: Trade sample is too small for robust learning. Hold parameters steady unless the last few months agree, prefer setup-specific tuning over global RSI shifts, and avoid large changes until trade count stabilizes.
- `2025-11` score=10.0000: No trades executed this month. Relax only the 15m trigger quality threshold or breakout volume filter modestly, keep the 1d macro filter intact, and keep shorts highly restrictive.
- `2025-12` score=15.0000: No trades executed this month. Relax only the 15m trigger quality threshold or breakout volume filter modestly, keep the 1d macro filter intact, and keep shorts highly restrictive.
