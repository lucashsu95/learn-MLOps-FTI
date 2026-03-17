---
marp: true
title: MLOps FTI
description: AAPL 5 日超額報酬預測實作
theme: mlops-theme
paginate: true
header: MLOps FTI 實作報告
footer: AAPL 5 日超額報酬預測
---

<!-- _class: lead -->
<!-- _paginate: skip -->
# MLOps FTI
## 從特徵到上線的完整閉環

預測標的：AAPL
預測任務：5 日超額報酬率（相對 SPY）

---

## 三件事

1. 我如何避免訓練與推論不一致（training-serving skew）
2. 我如何保證上線模型不是「退步模型」
3. 我如何把流程變成可持續運作的自動化系統

---

<!-- footer: Feature Pipeline -->

## 專案架構總覽

```text
yfinance（資料源）
     -> Feature Pipeline -> Hopsworks Feature Store
     -> Training Pipeline -> Hopsworks Model Registry
     -> Inference Pipeline（Modal 排程）
     -> Gradio UI（Hugging Face Spaces）
```

核心價值：一致性、可追蹤、自動化。

---

## Feature Pipeline：先把資料基礎打穩

Feature Pipeline 負責三件事：

1. 抓取 AAPL、SPY、QQQ、VIX 資料
2. 計算技術與市場脈絡特徵
3. 寫入 Feature Store 供訓練與推論共用

重點不是「多做特徵」，而是「把同一套特徵做對、做穩」。

---

## 共享模組：消除重複與漂移

```python
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        return df
```

同一個函數同時給訓練與推論使用，避免邏輯分叉。

---

<!-- footer: Training Pipeline -->

## 常數集中管理：改一次，全線生效

```python
RSI_PERIOD = 14
MACD_FAST_PERIOD = 12
TRAIN_TEST_SPLIT_RATIO = 0.8
MARKET_TICKERS = {"spy": "SPY", "qqq": "QQQ", "vix": "^VIX"}
```

這個設計讓維護成本更低，也降低「漏改」風險。

---

## Training Pipeline：三道品質關卡

1. 時序切分，不可 shuffle
2. Walk-Forward 驗證，檢查跨時間窗穩定性
3. Baseline Gating，未打敗 baseline 不上線

這三步一起做，避免指標好看但不可用。

---

## 時序驗證與守門邏輯

```python
split_idx = int(len(X) * TRAIN_TEST_SPLIT_RATIO)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

mae_pass = metrics["mae"] <= baseline_mae * (1 + tolerance)
rmse_pass = metrics["rmse"] < baseline_rmse
r2_pass = metrics["r2"] > baseline_r2
beats_baseline = mae_pass and rmse_pass and r2_pass
```

目標不是只追求單次高分，而是降低錯上線風險。

---

<!-- footer: Inference Pipeline -->

## Inference Pipeline：排程化執行

```python
@app.function(
        schedule=modal.Cron("0 22 * * 1-5"),
        timeout=300,
)
def scheduled_inference():
        return run_pipeline()
```

週一到週五 UTC 22:00（台灣早上 6 點）自動推論。

---

## CI/CD：讓流程可持續運作

1. CI：Ruff + pytest + 型別檢查
2. CD：app 更新自動部署到 Hugging Face Spaces
3. Retraining：每週六自動重訓（GitHub Actions 備援）

資料更新 -> 自動推論 -> 週期重訓 -> 模型更新 -> UI 展示

---

<!-- _class: lead -->
## Thank You

MLOps 的價值不只在模型分數，
更在於「可重複、可追蹤、可持續」。

