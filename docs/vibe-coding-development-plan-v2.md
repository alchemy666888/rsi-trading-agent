### 1. 整體架構（對應 developing-ai-agents-from-scratch-v2）
```
User Goal（例：「BTC/USDT 多週期策略，MaxDD < 25%」）
    ↓
Weekly Agent Loop (Perceive → Reason → Act → Evaluate → Upgrade)
    ├── Perceive: 載入 2023-01 ~ 2025-09 歷史 OHLCV，重採樣 15m / 1h / 4h / 1d，多週期特徵
    ├── Reason: Planner 產生多週期訊號與倉位權重（處理共振/衝突）
    ├── Act: 以「當週」資料做回測/紙上交易，產生交易序列
    ├── Evaluate: 計算 Weekly Score = Sharpe + WinRate − Drawdown Penalty − Costs
    └── Upgrade: 將本週 Lesson 寫入 Long-term Memory，微調參數並重新部署下一週
        ↓
Long-term Memory（按週存最優/最差片段）→ 下一週 Planner 自動載入
```
**Self-Improvement 閉環**：每週迴圈一次；低分週自動生成 Lesson，調整參數、風控或特徵權重，形成週級迭代飛輪。

---

### 2. 資料與工具（15m/1h/4h/1d 多週期）
- **資料窗口**：固定 2023-01-01 ~ 2025-09-30 的 BTC/USDT OHLCV。
- **重採樣**：原始較細資料 → 15m / 1h / 4h / 1d 四個 DataFrame；同週切片再對齊。
- **工具 Schema**（需更新）
  - `fetch_btc_data(timeframe: str, start: str, end: str)`: 拉取指定 timeframe & 日期範圍（ISO 日期字串），返回 OHLCV。
  - `resample_features(raw_df)`：生成四個 timeframe 資料並計算核心指標（RSI、MACD、EMA/MA、BB、波動率）。
  - `run_weekly_backtest(frames_by_tf, strategy_params, costs)`：對「單週」資料產生交易序列與指標。
  - `compute_weekly_score(metrics, costs)`：Sharpe + WinRate − λ_dd·MaxDD − λ_cost·Costs。
  - `persist_trace(week_id, strategy, metrics, score, decisions)`：寫入 traces/ 週級 JSON。
- **排程**：新增 `weekly_runner`（Cron/手動皆可）→ 每週一觸發上一週資料回測並升級參數。
- **預設成本模型**：手續費 0.04%/交易，滑點 0.02%/邊（可在工具輸入）。

---

### 3. 策略與訊號設計（多週期共振）
- **多週期共振**：若 15m/1h/4h 與 1d 方向一致，提高倉位權重；若衝突，縮小倉位或空倉。
- **參數版本化**：`strategy_v{ISO-week}`，每週保存完整參數組（RSI 門檻、MACD、均線長度、倉位上限、風控係數）。
- **風控上限**：
  - 單筆最大損失 < 0.5% 資本（以 ATR 或日內波動率調整）。
  - 單週最大回撤 > 25% 時，強制降低槓桿/倉位並標記為「高風險 Lesson」。
- **示例訊號邏輯（文件用，不是最終代碼）**：
```
signal = 0
if rsi_15m < rsi_buy and macd_1h > 0 and trend_1d_up:
    signal = +1 * weight_resonance
elif rsi_15m > rsi_sell and macd_1h < 0 and trend_1d_down:
    signal = -1 * weight_resonance
else:
    signal = 0 (conflict → reduce exposure)
```

---

### 4. Metacognition（Trading-Weekly）
- **Weekly Score**：
  - `score = w1*Sharpe + w2*WinRate - w3*MaxDD - w4*Costs`
  - 建議預設：w1=40, w2=30, w3=20, w4=10（可在配置調整）。
- **Lesson 存儲**：
  - 每週生成 JSON：`{week_id, score, lesson, best_fragments, worst_fragments, params}`。
  - 將得分前 3 與後 2 的策略片段寫入 Long-term Memory，供下週 Planner few-shot。
- **Prompt 注入**：Planner 系統提示自動附上 `Top-K Lessons` + `Known Pitfalls`（例如「MaxDD 因過度加倉 15m 共振」）。

---

### 5. 記憶與資料庫
- **表結構（建議）**：
  - `weekly_strategies(week_id PRIMARY KEY, params_json, metrics_json, score, lesson, costs, max_dd)`
  - `lesson_index(week_id, fragment, type ENUM('best','worst'))`
- **查詢策略**：
  - Planner 讀取最近 8 週的 `best` 片段 + 最近 4 週的 `worst` 片段。
  - 若 MaxDD 持續高於 30%，自動加大 w3 懲罰。

---

### 6. Observability（週級 Trace）
- 每週寫入 `traces/week_{YYYY-WW}.json`：包含 timeframe 指標、交易序列、score breakdown（Sharpe/WinRate/MaxDD/Costs）、升級決策。
- 若 Score < 門檻（默認 60），Trace 中標記 `"action": "rerun_with_tighter_risk"` 以便後續重跑。

---

### 7. 安全與邊界
- **HITL**：回測/模擬前詢問人工確認（仍保持 Simulation-only）。
- **無實盤金鑰**：ccxt 仍採匿名/公共端點；禁止寫入交易所。
- **週期重跑**：若資料缺口或 API 失敗，記錄並跳過該週，避免誤學習。

---

### 8. 測試計畫（文件驗收）
- 確認文件路徑為 `docs/vibe-coding-development-plan-v2.md` 並敘述週級流程。
- 檢查「2023-01 ~ 2025-09」固定窗口與 15m/1h/4h/1d 多週期已明確寫出。
- 驗證 Weekly Score 公式、權重與升級步驟已描述且連結到 Planner 更新。
- 確認工具/介面包含 timeframe + 日期範圍輸入，並移除泛用「2y」期間描述。

---

### 9. 假設
- 僅交易 BTC/USDT；所有回測為離線/紙上交易。
- 週期定義為 ISO 週（Mon-Sun），可依地區調整。
- 成本與權重可配置，預設值寫在工具/配置說明中。
