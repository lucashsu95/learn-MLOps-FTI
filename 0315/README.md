# MLOps FTI 專案：股票報酬預測系統

> 一個完整的 MLOps 練習專案，實作 FTI（Feature-Training-Inference）架構，用 XGBoost 預測股票超額報酬率。

## 專案架構

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Feature Pipeline                            │
│  yfinance → 技術指標 + 市場脈絡 + 基本面 → Hopsworks Feature Store   │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                            │
│  Feature Store → XGBoost (Regressor + Classifier) → Model Registry │
│  支援：時序切分、Walk-Forward 驗證、Baseline 守門、自動調參         │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       Inference Pipeline                            │
│  Modal Cron（每日交易日）→ 拉取最新資料 → 預測 → 寫回 Feature Store │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                            UI Layer                                 │
│              Hugging Face Spaces (Gradio) → 展示預測結果            │
└─────────────────────────────────────────────────────────────────────┘
```

## 技術棧

| 元件 | 技術 | 用途 |
|------|------|------|
| Feature Store | Hopsworks | 特徵版本管理、Model Registry |
| 訓練框架 | XGBoost | 回歸 + 分類雙模型 |
| 排程平台 | Modal | 每日推論、每週重訓 |
| UI 層 | Hugging Face Spaces | Gradio 介面展示 |
| CI/CD | GitHub Actions | Lint + Test + 自動部署 |

## 目前預測目標

- **預測標的**：AAPL（可透過 `TICKER` 環境變數切換）
- **預測目標**：5 日超額報酬率（AAPL 報酬 - SPY 報酬）
- **模型架構**：
  - **Regressor**：預測報酬幅度
  - **Classifier**：預測漲跌方向
  - 最終預測 = Classifier 方向 × Regressor 幅度（依 signal_threshold 過濾）

---

## 目前的特徵（27 個）

### 技術指標
- 移動平均：`ma_5`, `close_vs_ma20`, `close_vs_ma50`, `ma20_vs_ma50`
- 動量指標：`rsi_14`, `macd`, `macd_signal`, `macd_hist`
- 波動指標：`bb_width`, `bb_position`, `atr_14`
- 量價關係：`volume_ratio`
- 歷史報酬：`return_1d`, `return_5d`, `return_20d`

### 市場脈絡
- 大盤走勢：`spy_return_1d`, `spy_return_5d`, `qqq_return_1d`, `qqq_return_5d`
- 波動率：`vix_level`, `vix_change_1d`, `vix_vs_ma20`

### 基本面（靜態廣播）
- 估值：`pe_ratio`, `forward_pe`, `price_to_book`
- 獲利：`eps`, `profit_margin`, `revenue_growth`, `earnings_growth`

---

## R² 提升策略（重要）

> ⚠️ **現況分析**：預測超額報酬率本質上是高噪聲任務，R² 接近 0 是常見現象。這不代表模型無用，分類準確率和方向命中率更有實務價值。

### 短期可執行（低風險）

| 策略 | 做法 | 預期效果 |
|------|------|----------|
| 延長預測天期 | `TARGET_HORIZON_DAYS=10` 或 `20` | 長期報酬噪聲較低，R² 可能提升 |
| 改預測原始報酬 | `TARGET_MODE=raw` | 移除 SPY 相關噪聲，但失去 Alpha 意義 |
| 調整 signal_threshold | 減少預測筆數，只對高信心樣本評估 | MAE/R² 改善，但覆盖率下降 |

### 中期優化（中風險）

| 策略 | 做法 | 預期效果 |
|------|------|----------|
| 加入資金流特徵 | 抓取法人買賣超、融資融券、期貨持倉 | 對股價有領先效果 |
| 加入選擇權數據 | Put/Call Ratio、未平倉量變化 | 市場情緒指標 |
| 特徵交互項 | `rsi_14 * vix_level`, `return_5d / atr_14` | 捕捉非線性關係 |

### 長期改進（高風險）

| 策略 | 做法 | 預期效果 |
|------|------|----------|
| 時序模型 | 改用 LSTM/Transformer 處理序列依賴 | 可能捕捉更長期模式 |
| 多股票訓練 | 用同產業股票聯合訓練（遷移學習） | 資料量增加，泛化可能改善 |
| 新聞情緒 | NLP 抓取新聞情緒分數 | 事件驅動的價格變動 |

### 實務建議

對股票預測而言，**R² 不是唯一指標**。建議同時關注：

1. **方向命中率（Hit Rate）**：預測漲跌方向是否正確
2. **分類 F1 Score**：漲跌分類的平衡準確度
3. **Walk-Forward 穩定性**：跨時間窗的指標一致性
4. **夏普比率**：策略的風險調整後報酬（回測）

---

## 自動化現況（MLOps 閉環）

專案已具備完整自動化：

1. **自動調參**：`TimeSeriesSplit` + `RandomizedSearchCV`
2. **Baseline 守門**：MAE ≤ baseline、RMSE < baseline、R² > baseline
3. **Smoke Tests**：訓練後自動驗證輸入欄位與模型輸出
4. **版本管理**：上傳成功自動更新 `.env` 的 `MODEL_VERSION`
5. **每日推論**：Modal Cron（交易日 UTC 22:00）
6. **每週重訓**：Modal Cron + GitHub Actions 備用（週六 UTC 01:30）

---

## 快速開始

### 1. 安裝依賴

```bash
cd 0315
pip install -r requirements.txt
```

### 2. 設定環境變數

```bash
cp .env.example .env
# 編輯 .env，填入 Hopsworks API Key
```

### 3. 執行 Pipeline

```bash
# Feature Pipeline（抓資料、做特徵）
python feature_pipeline.py

# Training Pipeline（訓練模型）
python training_pipeline.py

# Inference Pipeline（本地測試）
python inference_pipeline.py
```

### 4. 部署排程（Modal）

```bash
# 部署每日推論
modal deploy inference_pipeline.py

# 部署每週重訓
modal deploy retraining_pipeline.py

# 手動觸發
modal run retraining_pipeline.py
```

---

## 必要帳號

| 服務 | 用途 | 免費額度 |
|------|------|----------|
| [Hopsworks](https://app.hopsworks.ai) | Feature Store + Model Registry | ✅ 夠用 |
| [Modal](https://modal.com) | Pipeline 排程 | ✅ 夠用 |
| [Hugging Face](https://huggingface.co) | UI 部署（Spaces） | ✅ 免費 |

---

## CI/CD 流程

- **CI**（Push/PR）：Ruff + Black + MyPy 檢查 → pytest 測試
- **CD**（Push to main）：自動部署到 HF Spaces
- **Retraining**（每週六 UTC 01:30）：GitHub Actions 備用重訓

---

## 專案結構

```
0315/
├── src/                    # 共享模組
│   ├── constants.py        # 魔術數字集中管理
│   ├── config.py           # 環境變數統一管理
│   ├── features.py         # 技術指標計算
│   └── utils.py            # 日誌與工具函數
├── tests/                  # pytest 測試
├── feature_pipeline.py     # 資料抓取 + 特徵工程
├── training_pipeline.py    # 模型訓練 + 評估
├── inference_pipeline.py   # 推論 + Modal 排程
├── retraining_pipeline.py  # 重訓入口
├── app.py                  # Gradio UI
└── .env                    # 環境變數（不進版控）
```

手動觸發一次：

```bash
modal run retraining_pipeline.py
```

每個專案都有這三個部分：

**Feature Pipeline** → **Training Pipeline** → **Inference Pipeline + UI**

資料流向：原始資料 → 特徵工程 → 存進 Feature Store (Hopsworks) → 訓練模型 → 存進 Model Registry → Inference 讀取 → Hugging Face / Streamlit 呈現

---

## 實做起點：三個帳號先開好

1. **[Hopsworks](https://app.hopsworks.ai)** — Feature Store + Model Registry（免費額度夠用）
2. **[Modal](https://modal.com)** — 排程執行 pipeline（免費額度夠用）
3. **[Hugging Face](https://huggingface.co)** — 部署 UI（Spaces 免費）

---
照著這個順序動手：

1. `feature_pipeline.py` — 抓資料、做特徵、寫進 Hopsworks Feature Store
2. `training_pipeline.py` — 從 Feature Store 讀取、訓練模型、存進 Model Registry
3. `inference_pipeline.py` — 定期（每天/每小時）拉新資料預測，存回 Hopsworks
4. `app.py` — HF Spaces 上的 Gradio / Streamlit UI，讀取最新預測顯示

```
yfinance (每日)          Alpha Vantage / yfinance (每季)
     ↓                              ↓
技術指標計算                    財報數據抓取
(RSI, MA20/50, MACD)        (EPS, P/E, Revenue)
     ↓                              ↓
          Hopsworks Feature Store
                    ↓
            training_pipeline.py
            (XGBoost Regressor)
                    ↓
          Hopsworks Model Registry
                    ↓
         inference_pipeline.py (Modal 每日排程)
                    ↓
           HF Spaces Gradio UI
```