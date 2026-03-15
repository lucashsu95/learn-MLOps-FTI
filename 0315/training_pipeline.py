"""
從 Hopsworks Feature Store 讀取特徵，
訓練 XGBoost 回歸模型，評估後存入 Model Registry。

使用方式：
    pip install xgboost hopsworks scikit-learn matplotlib
    python training_pipeline.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ── 設定區 ────────────────────────────────────────────────────────
TICKER = "AAPL"
HOPSWORKS_PROJECT  = os.environ.get("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY  = os.environ.get("HOPSWORKS_API_KEY")
FEATURE_GROUP_NAME = f"{TICKER.lower()}_stock_features"
FEATURE_GROUP_VERSION = int(os.environ.get("FEATURE_GROUP_VERSION"))
MODEL_NAME    = f"{TICKER.lower()}_xgb_regressor"
MODEL_VERSION = int(os.environ.get("MODEL_VERSION"))

# XGBoost 超參數
XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha":        0.1,   # L1 正則化
    "reg_lambda":       1.0,   # L2 正則化
    "random_state":     42,
    "n_jobs":           -1,
}

# 特徵欄位（與 feature_pipeline.py 對應）
FEATURE_COLS = [
    # 技術指標
    "ma_5",
    "close_vs_ma20", "close_vs_ma50", "ma20_vs_ma50",
    "bb_width", "bb_position",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "atr_14",
    "volume_ratio",
    "return_1d", "return_5d", "return_20d",
    # 基本面
    "pe_ratio", "forward_pe", "eps",
    "price_to_book", "profit_margin",
    "revenue_growth", "earnings_growth",
]
TARGET_COL = "target_next_close"
# ─────────────────────────────────────────────────────────────────


def _get_latest_feature_group(fs, name: str, min_version: int = 1, max_version: int = 20):
    """取得指定名稱的最新可用 Feature Group 版本。"""
    last_error = None
    for version in range(max_version, min_version - 1, -1):
        try:
            fg = fs.get_feature_group(name, version=version)
            if fg is None:
                continue
            return fg, version
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"找不到可用的 Feature Group: {name} v{min_version}~v{max_version}") from last_error


def _validate_hopsworks_config() -> None:
    """確保必要環境變數存在，避免用假值誤連線。"""
    if not HOPSWORKS_PROJECT:
        raise ValueError("缺少 HOPSWORKS_PROJECT，請在 .env 或系統環境變數設定。")
    if not HOPSWORKS_API_KEY:
        raise ValueError("缺少 HOPSWORKS_API_KEY，請在 .env 或系統環境變數設定。")


# ── Step 1：讀取 Feature Store ────────────────────────────────────
def load_features_from_hopsworks() -> pd.DataFrame:
    print("[1/5] 從 Hopsworks Feature Store 讀取資料...")
    try:
        import hopsworks
    except ImportError:
        raise ImportError("請先安裝：pip install hopsworks")

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY
    )
    fs = project.get_feature_store()
    fg, version = _get_latest_feature_group(fs, FEATURE_GROUP_NAME, min_version=1)
    df = fg.read()
    print(f"    ✓ 讀取 {len(df)} 筆資料（{FEATURE_GROUP_NAME} v{version}）")
    return df


def load_features_from_local(csv_path: str = None) -> pd.DataFrame:
    """
    Hopsworks 還沒設定時的替代方案：
    直接執行 feature_pipeline.py 取得 df，或從 CSV 讀取。
    """
    if csv_path and os.path.exists(csv_path):
        print(f"[1/5] 從本地 CSV 讀取：{csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("[1/5] 本地模式：直接執行 feature_pipeline 取得資料...")
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from feature_pipeline import main as run_feature_pipeline
        df = run_feature_pipeline()

    print(f"    ✓ 讀取 {len(df)} 筆資料")
    return df


# ── Step 2：特徵工程 & 資料切分 ───────────────────────────────────
def prepare_data(df: pd.DataFrame):
    print("[2/5] 準備訓練資料...")

    # 確保按日期排序（時序資料不能 shuffle！）
    df = df.sort_values("date").reset_index(drop=True)

    # 只保留有值的特徵欄位
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"    ⚠ 以下特徵欄位不存在，跳過: {missing}")

    X = df[available_features].copy()
    y = df[TARGET_COL].copy()

    # 填補基本面 NaN（靜態欄位偶爾為 None）
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # 時序切分（後 20% 當 test，不 shuffle）
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"    ✓ 特徵數: {len(available_features)}")
    print(f"    ✓ Train: {len(X_train)} 筆  |  Test: {len(X_test)} 筆")
    print(f"    ✓ 訓練期間: {df['date'].iloc[0]} ~ {df['date'].iloc[split_idx-1]}")
    print(f"    ✓ 測試期間: {df['date'].iloc[split_idx]} ~ {df['date'].iloc[-1]}")

    return X_train, X_test, y_train, y_test, available_features, df


# ── Step 3：訓練 ──────────────────────────────────────────────────
def train_model(X_train, y_train):
    print("[3/5] 訓練 XGBoost 模型...")
    from xgboost import XGBRegressor

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )
    print(f"    ✓ 訓練完成（n_estimators={XGB_PARAMS['n_estimators']}）")
    return model


# ── Step 4：評估 ──────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, feature_names, df, split_idx):
    print("[4/5] 評估模型...")
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # 百分比誤差

    metrics = {
        "mae":  round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2":   round(float(r2), 4),
        "mape": round(float(mape), 4),
    }

    print(f"    MAE  = {mae:.4f}  （平均絕對誤差，單位：美元）")
    print(f"    RMSE = {rmse:.4f}")
    print(f"    R²   = {r2:.4f}  （1.0 最佳）")
    print(f"    MAPE = {mape:.2f}%  （平均百分比誤差）")

    # 繪圖
    _plot_results(y_test, y_pred, feature_names, model, df, split_idx)

    return metrics, y_pred


def _plot_results(y_test, y_pred, feature_names, model, df, split_idx):
    """產生兩張圖：預測走勢 + 特徵重要性"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 圖一：預測 vs 實際
    ax = axes[0]
    dates = df["date"].iloc[split_idx:split_idx + len(y_test)].values
    ax.plot(dates, y_test.values, label="實際收盤價", color="#2196F3", linewidth=1.5)
    ax.plot(dates, y_pred,        label="預測收盤價", color="#FF5722", linewidth=1.5, linestyle="--")
    ax.set_title(f"{TICKER} 預測 vs 實際（測試集）")
    ax.set_xlabel("日期")
    ax.set_ylabel("收盤價（USD）")
    ax.legend()
    # 只顯示少量 x 刻度避免擠在一起
    step = max(1, len(dates) // 6)
    ax.set_xticks(dates[::step])
    ax.tick_params(axis="x", rotation=30)

    # 圖二：特徵重要性（Top 15）
    ax2 = axes[1]
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]
    ax2.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color="#4CAF50"
    )
    ax2.set_title("特徵重要性 Top 15")
    ax2.set_xlabel("Importance Score")

    plt.tight_layout()
    chart_path = "training_results.png"
    plt.savefig(chart_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    ✓ 圖表已儲存：{chart_path}")
    return chart_path


# ── Step 5：儲存至 Model Registry ─────────────────────────────────
def save_model_to_hopsworks(model, metrics, feature_names):
    print("[5/5] 儲存模型至 Hopsworks Model Registry...")
    import joblib

    # 先存本地
    model_dir = f"model_{TICKER.lower()}"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)

    # 存 metadata
    metadata = {
        "ticker":        TICKER,
        "feature_cols":  feature_names,
        "xgb_params":    XGB_PARAMS,
        "metrics":       metrics,
        "trained_at":    datetime.now().isoformat(),
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"    ✓ 模型已存至本地：{model_dir}/")

    # 上傳至 Hopsworks
    try:
        import hopsworks
        project = hopsworks.login(
            project=HOPSWORKS_PROJECT,
            api_key_value=HOPSWORKS_API_KEY
        )
        mr = project.get_model_registry()

        model_obj = mr.sklearn.create_model(
            name=MODEL_NAME,
            version=MODEL_VERSION,
            metrics=metrics,
            description=f"{TICKER} next-day close price prediction (XGBoost)",
            input_example=None,
        )
        model_obj.save(model_dir)
        print(f"    ✓ 已上傳至 Hopsworks Model Registry：{MODEL_NAME} v{MODEL_VERSION}")

    except Exception as e:
        print(f"    ⚠ Hopsworks 上傳失敗（本地模式可忽略）: {e}")
        print(f"    模型保留在本地：{model_dir}/")


# ── 主流程 ────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print(f"  Stock Training Pipeline  |  Ticker: {TICKER}")
    print("=" * 55)

    # 先檢查環境變數，避免誤用預設假值。
    _validate_hopsworks_config()

    # 讀資料：優先嘗試 Hopsworks，失敗則走本地
    try:
        df = load_features_from_hopsworks()
    except Exception as e:
        print(f"    ⚠ Hopsworks 連線失敗，改用本地模式: {e}")
        df = load_features_from_local()

    X_train, X_test, y_train, y_test, feature_names, df_sorted = prepare_data(df)
    split_idx = int(len(df) * 0.8)

    model = train_model(X_train, y_train)
    metrics, y_pred = evaluate_model(
        model, X_test, y_test, feature_names, df_sorted, split_idx
    )
    save_model_to_hopsworks(model, metrics, feature_names)

    print("\n✅ Training Pipeline 執行完畢")
    print(f"   最終 MAPE：{metrics['mape']}%  |  R²：{metrics['r2']}")
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
