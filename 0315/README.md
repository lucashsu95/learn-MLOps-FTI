## 

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