### 1. Core Foundations: AI Agent 核心架構（Brain + Environment + Sensors + Actuators + Loop）
**解釋**：  
AI Agent 不是普通 LLM Chatbot，而是具備**自主感知-推理-行動**能力的系統。四大支柱缺一不可：  
- **Brain（LLM）**：負責規劃、決策、評估（例如 GPT-4o、Claude 3.5 等）。  
- **Environment**：Agent 運作的外部空間（例如航班預訂系統、企業 CRM、本地 OS）。  
- **Sensors**：感知機制（讀取 API 回應、資料庫查詢、使用者輸入）。  
- **Actuators**：改變環境的機制（執行 Python 腳本、發送 Email、寫入資料庫）。  

**Agent Loop**：**Perceive（感知）→ Reason（推理）→ Act（行動）** 不斷循環，直到目標達成。  

**Trading Agent Loop（歷史資料 2023-01 ~ 2025-09，週期：15m / 1h / 4h / 1d）**：
1) **Perceive**：載入 2023-01 至 2025-09 的 OHLCV，分別重採樣 15 分鐘、1 小時、4 小時、1 天；生成技術特徵與風險指標，存入當週訓練批次。  
2) **Reason**：基於多週期特徵產生交易訊號與倉位（例如共振/沖突處理、風控上限）。  
3) **Act**：對當週資料進行回測或紙上交易，產生實際交易序列。  
4) **Evaluate（Weekly Score）**：每週計算學習分數（例如 Sharpe、勝率、最大回撤懲罰、成交滑點成本），形成「本週教訓」。  
5) **Upgrade Strategy Weekly**：根據 Weekly Score 自動調整策略（參數微調、特徵重要度、風控門檻），並在下一週重新部署；持續寫入 Long-term Memory 以避免重複錯誤。  

**例子**：旅行規劃 Agent 先感知使用者需求（Sensor），用 LLM 推理出「先查航班、再訂飯店」（Reason），再呼叫訂票 API（Act）。  
**細微差別與邊緣情境**：如果 Environment 是動態的（例如股市即時變化），Loop 必須包含「失敗重試」機制；若 Sensors 過載（太多 API 同時回應），會導致上下文爆炸。  
**為何重要**：這是所有 Agent 的「骨架」，沒有它就只是「會說話的提示詞」。  
**對自我改進的貢獻**：Loop 本身就是「試錯學習」的基礎，讓 Agent 能在每次循環中累積經驗。

---

### 2. Agentic Frameworks & Tooling（MAF、Semantic Kernel、AutoGen、MCP、A2A）
**解釋**：  
不要從零寫一切，必須使用框架加速開發。筆記重點推薦 **Microsoft Agent Framework (MAF)** 作為統一 orchestrator，搭配 Azure AI Foundry、Semantic Kernel、AutoGen。  
標準技術棧：Python 3.12+ / .NET + MAF + **MCP（Model Context Protocol）**（模型與工具/資料的標準連接協定）+ **A2A（Agent-to-Agent）**（多 Agent 間溝通標準）。  

**例子**：用 MAF 建立一個「旅行 Agent」，MCP 讓它輕鬆呼叫外部航班 API，A2A 讓它與「飯店子 Agent」對話。  
**細微差別**：MCP 解決「工具格式不統一」的痛點；A2A 讓多 Agent 像微服務一樣協作。  
**為何重要**：從「腳本級」升級到「可擴展、可維護」的企業級 Agent。  
**對自我改進的貢獻**：框架內建的狀態管理（Threads）讓 Agent 能「暫停-恢復」，跨會話累積學習。

---

### 3. Context Engineering（上下文工程） vs. Prompt Engineering
**解釋**：  
Prompt Engineering 是「靜態指令」；Context Engineering 是「動態資訊管理」，解決上下文視窗（token limit）瓶頸。  
**5 種必須管理的 Context**：  
1. Instructions（系統提示、few-shot、工具描述）  
2. Knowledge（RAG 檢索的知識）  
3. Tools（工具定義 + 執行結果）  
4. Conversation History（對話歷史）  
5. User Preferences（長期使用者偏好）  

**管理策略**（4 大技巧）：  
- Agent Scratchpad（臨時筆記本，不塞進主上下文）  
- Long-term Memories（資料庫儲存摘要與偏好）  
- Compressing Context（自動摘要舊歷史）  
- Sandbox Environments（大 CSV 丟給 code interpreter，只回傳結果）  

**例子**：處理 10MB 銷售報表時，不要把 CSV 塞進 LLM，而是 Sandbox 計算後只回「總營收 125 萬」。  
**邊緣情境**：超長多輪對話（>100 輪）若不壓縮，token 成本暴增 10 倍以上。  
**對自我改進的貢獻**：Long-term Memories + Scratchpad 是「記憶與學習」的核心，讓 Agent 跨會話記住「上次我犯的錯」。

---

### 4. Core Agentic Design Patterns（四大設計模式）
**解釋**：這是 Agent「如何思考與運作」的藍圖，必須全部精通：  

**A. Tool Use Pattern**：LLM 透過 Function Calling 呼叫外部工具（Knowledge Tools + Action Tools）。關鍵：工具 schema 文件必須極其清晰。  
**B. Planning Pattern**：任務分解 + 迭代規劃（失敗時動態調整）。輸出必須是結構化 JSON。  
**C. Multi-Agent Pattern**：專門化微 Agent + Orchestrator（避免「上帝 Agent」幻覺）。  
**D. Metacognition Pattern（自我反思）**：Agent 先產出答案 → 呼叫 Reflection Tool 打分 → 若分數低就重寫。這是**自我改進**的最直接機制。  

**Trading-Weekly Metacognition**：每週收盤後（或固定週期），以 2023-01 ~ 2025-09 的歷史窗格與當週交易結果計算 Lesson Score（Sharpe + 胜率 − 回撤懲罰 − 交易成本），並將得分高/低的策略片段寫入 Long-term Memory；下一週自動用得分引導的參數優化或策略升級。  

**例子**：Multi-Agent 旅行系統（航班 Agent + 飯店 Agent），Planner Agent 統籌，Metacognition Agent 檢查「這行程是否符合預算？」  
**細微差別**：單 Agent 適合簡單任務；Multi-Agent + Metacognition 才具備真正「自我學習」能力。  
**對自我改進的貢獻**：Metacognition + Planning = 自我修正迴路；Multi-Agent 讓每個子 Agent 都能獨立進化。

---

### 5. Building Trustworthy & Secure Agents（可信與安全）
**解釋**：Agent 能執行真實動作，風險極高（Prompt Injection、未授權操作）。  
**系統訊息框架**：用 Meta System Message（讓另一個 LLM 產生最佳系統提示）+ 明確「不能做什麼」的邊界。  
**安全最佳實務**：  
- Human-in-the-Loop（HITL）：高風險動作（轉帳、發企業 Email）必須人工批准。  
- Keyless Authentication（Microsoft Entra ID 等雲端身份驗證，避免 .env 硬編碼）。  

**例子**：金融 Agent 絕對不能直接退款，必須先呼叫 HITL 等待主管點頭。  
**邊緣情境**：如果不做邊界定義，Agent 可能被越獄執行惡意 SQL。  
**對自我改進的貢獻**：安全機制讓 Agent 敢大膽實驗（因為有「安全網」），才能持續自我優化。

---

### 6. Advanced Framework Features（MAF 高級功能）
**解釋**：從原型到企業應用必備：  
- **Threads**：管理狀態，讓 Agent 能「暫停-等待 API-恢復」。  
- **Agent Middleware**：在 LLM 決策與工具執行中間插入程式碼（記錄、權限檢查、隱藏參數注入）。  
- **Event Triggers**：WorkflowStartedEvent、ExecutorInvokeEvent 等，精準追蹤 Agent 思考過程。  

**例子**：長時間航班查詢時，Thread 可讓 Agent 先回「正在查詢…」，等 API 回來再繼續。  
**為何重要**：這些功能讓 Agent 具備「企業級可靠性」。  
**對自我改進的貢獻**：Middleware 可自動記錄每次反思結果，餵給 Long-term Memory 進行學習。

---

### 7. Production, Observability, & Evaluation（生產化、可觀測性、評估）
**解釋**：Agent 自主運作，必須有「X 光」視野。  
- **OpenTelemetry**：Trace（完整端到端流程）+ Spans（每個步驟：LLM Reasoning、Tool Call 等）。  
**4 大關鍵指標**：  
1. Latency（總耗時）  
2. Cost Tracking（token 消耗）  
3. Request Errors/Fallbacks（多 LLM 供應商備援）  
4. Accuracy（用 RAGAS、LLM Guard 自動評分）  

**例子**：若某 Trace 的「Reflection Span」分數長期 < 80%，就自動觸發模型升級或提示優化。  
**邊緣情境**：高並發時，Observability 能即時發現「某子 Agent 陷入無限反思迴圈」。  
**對自我改進的貢獻**：這是「外部監督機制」，讓開發者根據真實指標持續優化 Agent 的自我改進能力。

---

### 知識點之間的關係與互聯架構（如何形成 Self-Improving 閉環）
這些領域**不是孤立的**，而是層層相依、形成**自我改進飛輪**：

1. **Foundation（1）是地基** → 所有其他知識都在其上運行（Loop 是最基本的「學習引擎」）。  
2. **Frameworks（2）+ Advanced Features（6）提供「骨骼與神經」** → 讓 Context Engineering（3）、Design Patterns（4）、Security（5）能真正落地執行。  
3. **Context Engineering（3）解決「記憶瓶頸」** → 直接支援 Long-term Memories，讓 Metacognition（4D）能跨會話反思「我上次為什麼失敗」。  
4. **Design Patterns（4）是「大腦算法」**：  
   - Tool Use + Planning = 執行力  
   - Multi-Agent = 可擴展性  
   - **Metacognition = 自我改進核心**（這是唯一能讓 Agent「自己變聰明」的模式）。  
5. **Security（5）是「安全閥」** → 讓 Agent 敢大膽使用 Planning + Metacognition（否則一出錯就全部停擺）。  
6. **Production & Observability（7）是「外部教練」** → 透過 Trace 與自動評分，持續餵資料給 Long-term Memories 和 Metacognition，讓 Agent 自動迭代（例如發現「Planning 常失敗」→ 自動優化 planner prompt）。  

**整體 Self-Improving 閉環**（最重要關係）：  
**Perceive（Sensors + Context） → Reason（Planning + Metacognition） → Act（Tools + Multi-Agent） → Evaluate（Observability + Reflection） → Update Memory → 下一輪更強**  

這就是為什麼筆記強調「Metacognition + Long-term Memories + Threads + Observability」組合：  
- 沒有 Metacognition → Agent 永遠不會自己改錯  
- 沒有 Long-term Memories → 每次都是「失憶新生」  
- 沒有 Observability → 開發者無法知道該改哪裡  
- 沒有 Frameworks → 以上全部無法高效實現  

**進階啟示**：真正頂級的 Self-Improving Agent 會同時運行**兩個層級**：  
- **Operational Loop**（日常任務）  
- **Meta-Learning Loop**（每 100 次任務後，用 Metacognition + Observability 自動調整自身系統提示、工具 schema、甚至切換模型）  
