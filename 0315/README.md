## 

### 自動化現況（MLOps 閉環）

目前專案已具備以下自動化能力：

1. 自動調參與時序驗證：`training_pipeline.py`
2. 自動 baseline 守門：未達標不會上傳新模型
3. 自動 smoke tests：訓練後會檢查輸入欄位對齊與模型輸出有效性
4. 自動更新版本：上傳成功後會自動把 `.env` 的 `MODEL_VERSION` 更新為新版本
5. 自動推論排程：`inference_pipeline.py` 已有 Modal Cron 排程（交易日）
6. 自動重訓排程：`retraining_pipeline.py` 提供每週重訓入口（可 deploy 到 Modal）

### 部署自動重訓

```bash
modal deploy retraining_pipeline.py
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