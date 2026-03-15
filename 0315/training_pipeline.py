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
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")
load_dotenv()

# ── 設定區 ────────────────────────────────────────────────────────
TICKER = os.environ.get("TICKER", "AAPL").upper()
HOPSWORKS_PROJECT  = os.environ.get("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY  = os.environ.get("HOPSWORKS_API_KEY")
FEATURE_GROUP_NAME = f"{TICKER.lower()}_stock_features"


def _get_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} 必須是整數，收到: {raw}") from e


FEATURE_GROUP_VERSION = _get_int_env("FEATURE_GROUP_VERSION", 1)
MODEL_NAME    = f"{TICKER.lower()}_xgb_regressor"
MODEL_VERSION = _get_int_env("MODEL_VERSION", 1)
TARGET_HORIZON_DAYS = _get_int_env("TARGET_HORIZON_DAYS", 5)
SIGNAL_THRESHOLD = float(os.environ.get("SIGNAL_THRESHOLD", "0.58"))
REGISTER_IF_BEAT_BASELINE = os.environ.get("REGISTER_IF_BEAT_BASELINE", "1") == "1"
FORCE_REGISTER = os.environ.get("FORCE_REGISTER", "0") == "1"
WALK_FORWARD_SPLITS = _get_int_env("WALK_FORWARD_SPLITS", 5)
TARGET_MODE = os.environ.get("TARGET_MODE", "excess_spy").strip().lower()
HALF_LIFE_DAYS = float(os.environ.get("HALF_LIFE_DAYS", "252"))
MAE_TOLERANCE_PCT = float(os.environ.get("MAE_TOLERANCE_PCT", "0.01"))

if TARGET_MODE not in {"raw", "excess_spy"}:
    raise ValueError(f"TARGET_MODE 僅支援 raw / excess_spy，收到: {TARGET_MODE}")

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
    # 市場脈絡
    "spy_return_1d", "spy_return_5d",
    "qqq_return_1d", "qqq_return_5d",
    "vix_level", "vix_change_1d", "vix_vs_ma20",
]
if TARGET_MODE == "raw":
    TARGET_COL = f"target_{TARGET_HORIZON_DAYS}d_return"
else:
    TARGET_COL = f"target_excess_spy_{TARGET_HORIZON_DAYS}d_return"
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
    fg, version = _get_latest_feature_group(
        fs,
        FEATURE_GROUP_NAME,
        min_version=FEATURE_GROUP_VERSION,
    )
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
        print("[1/5] 本地模式：執行特徵計算（不寫入 Hopsworks）...")
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from feature_pipeline import (
            fetch_price_data,
            add_technical_indicators,
            fetch_fundamental_data,
            merge_fundamentals,
            clean_dataframe,
            TICKER as FEATURE_TICKER,
            PERIOD as FEATURE_PERIOD,
        )

        df = fetch_price_data(FEATURE_TICKER, FEATURE_PERIOD)
        df = add_technical_indicators(df)
        fundamentals = fetch_fundamental_data(FEATURE_TICKER)
        df = merge_fundamentals(df, fundamentals)
        df = clean_dataframe(df)

    print(f"    ✓ 讀取 {len(df)} 筆資料")
    return df


# ── Step 2：特徵工程 & 資料切分 ───────────────────────────────────
def prepare_data(df: pd.DataFrame):
    print("[2/5] 準備訓練資料...")

    # 確保按日期排序（時序資料不能 shuffle！）
    df = df.sort_values("date").reset_index(drop=True)

    # 某些 Feature Group 版本可能沒有 target 欄位，這裡用 close/benchmark 自動重建。
    if TARGET_COL not in df.columns:
        if "close" not in df.columns:
            raise KeyError(f"資料缺少 {TARGET_COL}，且無法用 close 重建目標欄位。")

        raw_target_col = f"target_{TARGET_HORIZON_DAYS}d_return"
        if raw_target_col in df.columns:
            raw_target = df[raw_target_col]
        else:
            raw_target = df["close"].shift(-TARGET_HORIZON_DAYS) / df["close"] - 1

        if TARGET_MODE == "excess_spy":
            benchmark_col = f"spy_return_{TARGET_HORIZON_DAYS}d"
            if benchmark_col not in df.columns:
                raise KeyError(f"資料缺少 {benchmark_col}，無法建立超額報酬目標。")
            print(f"    ⚠ 找不到 {TARGET_COL}，改用 TSLA-{benchmark_col} 建立超額報酬目標。")
            df[TARGET_COL] = raw_target - df[benchmark_col].fillna(0.0)
        else:
            print(f"    ⚠ 找不到 {TARGET_COL}，改用 {TARGET_HORIZON_DAYS} 日報酬率自動建立。")
            df[TARGET_COL] = raw_target

    # 去除沒有 target 的尾端資料。
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

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
    if TARGET_MODE == "excess_spy":
        print(f"    ✓ 預測目標: {TARGET_HORIZON_DAYS} 日超額報酬率（TSLA - SPY）")
    else:
        print(f"    ✓ 預測目標: {TARGET_HORIZON_DAYS} 日報酬率（{TARGET_COL}）")

    return X_train, X_test, y_train, y_test, available_features, df


# ── Step 3：訓練 ──────────────────────────────────────────────────
def train_model(X_train, y_train):
    print("[3/5] 訓練 XGBoost 模型...")
    from xgboost import XGBRegressor, XGBClassifier

    # 對近期資料給較高權重，強化當前市場狀態的學習
    if HALF_LIFE_DAYS > 0:
        steps = np.arange(len(X_train), dtype=float)
        distances = (len(X_train) - 1) - steps
        sample_weights = np.power(0.5, distances / HALF_LIFE_DAYS)
    else:
        sample_weights = np.ones(len(X_train), dtype=float)

    base_reg = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    param_dist = {
        "n_estimators": [200, 300, 500, 800, 1200],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 8, 10],
        "reg_alpha": [0.0, 0.1, 0.3, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        estimator=base_reg,
        param_distributions=param_dist,
        n_iter=18,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train, sample_weight=sample_weights)

    reg_params = dict(search.best_params_)
    reg_params["objective"] = "reg:squarederror"
    reg_params["random_state"] = 42
    reg_params["n_jobs"] = -1

    reg_model = XGBRegressor(**reg_params)
    reg_model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )

    y_cls_train = (y_train > 0).astype(int)
    cls_model = XGBClassifier(
        objective="binary:logistic",
        random_state=42,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
    )
    cls_model.fit(X_train, y_cls_train, sample_weight=sample_weights, verbose=False)

    print("    ✓ TimeSeriesSplit 調參完成")
    print(f"    ✓ 最佳參數: {search.best_params_}")
    print(f"    ✓ CV 最佳 MAE: {-search.best_score_:.4f}")
    print(f"    ✓ 近期加權半衰期: {HALF_LIFE_DAYS} 天")
    return {
        "regressor": reg_model,
        "classifier": cls_model,
    }, {
        "regressor_best_params": search.best_params_,
        "regressor_cv_best_mae": round(float(-search.best_score_), 6),
        "signal_threshold": SIGNAL_THRESHOLD,
        "half_life_days": HALF_LIFE_DAYS,
    }


# ── Step 4：評估 ──────────────────────────────────────────────────
def evaluate_model(models, X_test, y_test, feature_names, df, split_idx):
    print("[4/5] 評估模型...")
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    reg_model = models["regressor"]
    cls_model = models["classifier"]

    y_reg_pred = reg_model.predict(X_test)
    y_up_prob = cls_model.predict_proba(X_test)[:, 1]
    y_cls_pred = (y_up_prob >= 0.5).astype(int)

    y_pred = np.zeros_like(y_reg_pred)
    long_mask = y_up_prob >= SIGNAL_THRESHOLD
    short_mask = y_up_prob <= (1 - SIGNAL_THRESHOLD)
    y_pred[long_mask] = np.abs(y_reg_pred[long_mask])
    y_pred[short_mask] = -np.abs(y_reg_pred[short_mask])

    baseline_pred = np.zeros_like(y_test.values, dtype=float)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    baseline_mae  = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    baseline_r2   = r2_score(y_test, baseline_pred)
    denom = np.maximum(np.abs(y_test), 1e-6)
    mape = np.mean(np.abs((y_test - y_pred) / denom)) * 100
    hit_rate = np.mean(np.sign(y_test.values) == np.sign(y_pred)) * 100
    baseline_hit_rate = np.mean(np.sign(y_test.values) == 0) * 100
    cls_acc = accuracy_score((y_test.values > 0).astype(int), y_cls_pred)
    cls_f1 = f1_score((y_test.values > 0).astype(int), y_cls_pred)

    metrics = {
        "mae":  round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2":   round(float(r2), 4),
        "mape": round(float(mape), 4),
        "hit_rate": round(float(hit_rate), 2),
        "baseline_mae": round(float(baseline_mae), 4),
        "baseline_rmse": round(float(baseline_rmse), 4),
        "baseline_r2": round(float(baseline_r2), 4),
        "baseline_hit_rate": round(float(baseline_hit_rate), 2),
        "cls_accuracy": round(float(cls_acc), 4),
        "cls_f1": round(float(cls_f1), 4),
        "signal_threshold": round(float(SIGNAL_THRESHOLD), 4),
    }

    print(f"    MAE  = {mae:.4f}  （baseline: {baseline_mae:.4f}）")
    print(f"    RMSE = {rmse:.4f}  （baseline: {baseline_rmse:.4f}）")
    print(f"    R²   = {r2:.4f}  （baseline: {baseline_r2:.4f}）")
    print(f"    MAPE = {mape:.2f}%  （報酬率百分比誤差）")
    print(f"    方向命中率 = {hit_rate:.2f}%  （baseline: {baseline_hit_rate:.2f}%）")
    print(f"    分類準確率 = {cls_acc:.4f}  |  F1 = {cls_f1:.4f}")

    # 繪圖
    _plot_results(y_test, y_pred, feature_names, reg_model, df, split_idx)

    return metrics, y_pred


def evaluate_walk_forward(df: pd.DataFrame, feature_names: list, reg_params: dict):
    """多視窗 walk-forward 評估，用來觀察時序穩定性。"""
    print("[4.5/5] Walk-forward 多視窗評估...")
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    X_all = df[feature_names].copy()
    y_all = df[TARGET_COL].copy()

    for col in X_all.columns:
        if X_all[col].isna().any():
            X_all[col] = X_all[col].fillna(X_all[col].median())

    tscv = TimeSeriesSplit(n_splits=WALK_FORWARD_SPLITS)
    fold_mae = []
    fold_rmse = []
    fold_r2 = []

    clean_params = {
        "n_estimators": reg_params["n_estimators"],
        "max_depth": reg_params["max_depth"],
        "learning_rate": reg_params["learning_rate"],
        "subsample": reg_params["subsample"],
        "colsample_bytree": reg_params["colsample_bytree"],
        "min_child_weight": reg_params["min_child_weight"],
        "reg_alpha": reg_params["reg_alpha"],
        "reg_lambda": reg_params["reg_lambda"],
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
    }

    for i, (train_idx, test_idx) in enumerate(tscv.split(X_all), start=1):
        X_tr, X_te = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_tr, y_te = y_all.iloc[train_idx], y_all.iloc[test_idx]

        model = XGBRegressor(**clean_params)
        model.fit(X_tr, y_tr, verbose=False)
        pred = model.predict(X_te)

        mae = mean_absolute_error(y_te, pred)
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
        r2 = r2_score(y_te, pred)
        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)
        print(f"    Fold {i}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    wf_metrics = {
        "wf_mae_mean": round(float(np.mean(fold_mae)), 4),
        "wf_mae_std": round(float(np.std(fold_mae)), 4),
        "wf_rmse_mean": round(float(np.mean(fold_rmse)), 4),
        "wf_r2_mean": round(float(np.mean(fold_r2)), 4),
        "wf_r2_std": round(float(np.std(fold_r2)), 4),
        "wf_splits": WALK_FORWARD_SPLITS,
    }

    print(f"    ✓ WF 平均 MAE: {wf_metrics['wf_mae_mean']:.4f} ± {wf_metrics['wf_mae_std']:.4f}")
    print(f"    ✓ WF 平均 R² : {wf_metrics['wf_r2_mean']:.4f} ± {wf_metrics['wf_r2_std']:.4f}")
    return wf_metrics


def _plot_results(y_test, y_pred, feature_names, model, df, split_idx):
    """產生兩張圖：預測走勢 + 特徵重要性"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 圖一：預測 vs 實際
    ax = axes[0]
    dates = df["date"].iloc[split_idx:split_idx + len(y_test)].values
    y_label = f"{TARGET_HORIZON_DAYS} 日超額報酬率" if TARGET_MODE == "excess_spy" else f"{TARGET_HORIZON_DAYS} 日報酬率"
    ax.plot(dates, y_test.values, label=f"實際 {y_label}", color="#2196F3", linewidth=1.5)
    ax.plot(dates, y_pred,        label=f"預測 {y_label}", color="#FF5722", linewidth=1.5, linestyle="--")
    ax.set_title(f"{TICKER} {y_label}：預測 vs 實際（測試集）")
    ax.set_xlabel("日期")
    ax.set_ylabel("報酬率")
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
def save_model_to_hopsworks(models, metrics, feature_names, tuning_info, should_register: bool):
    print("[5/5] 儲存模型至 Hopsworks Model Registry...")
    import joblib

    # 先存本地
    model_dir = f"model_{TICKER.lower()}"
    os.makedirs(model_dir, exist_ok=True)

    reg_model_path = os.path.join(model_dir, "reg_model.pkl")
    cls_model_path = os.path.join(model_dir, "cls_model.pkl")
    # 相容舊流程：保留 model.pkl 指向回歸模型
    legacy_model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(models["regressor"], reg_model_path)
    joblib.dump(models["classifier"], cls_model_path)
    joblib.dump(models["regressor"], legacy_model_path)

    # 存 metadata
    metadata = {
        "ticker":        TICKER,
        "feature_cols":  feature_names,
        "xgb_params":    {
            "regressor": models["regressor"].get_params(),
            "classifier": models["classifier"].get_params(),
        },
        "target_col":    TARGET_COL,
        "target_horizon_days": TARGET_HORIZON_DAYS,
        "target_mode":   TARGET_MODE,
        "target_benchmark": "SPY" if TARGET_MODE == "excess_spy" else None,
        "metrics":       metrics,
        "tuning":        tuning_info,
        "trained_at":    datetime.now().isoformat(),
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"    ✓ 模型已存至本地：{model_dir}/")

    if not should_register and not FORCE_REGISTER:
        print("    ⚠ 守門啟用：本次模型未優於 baseline，略過 Model Registry 上傳。")
        return

    # 上傳至 Hopsworks
    try:
        import hopsworks
        project = hopsworks.login(
            project=HOPSWORKS_PROJECT,
            api_key_value=HOPSWORKS_API_KEY
        )
        mr = project.get_model_registry()

        saved_version = None
        for offset in range(5):
            candidate_version = MODEL_VERSION + offset
            try:
                model_obj = mr.sklearn.create_model(
                    name=MODEL_NAME,
                    version=candidate_version,
                    metrics=metrics,
                    description=f"{TICKER} {TARGET_HORIZON_DAYS}-day return prediction (XGBoost)",
                    input_example=None,
                )
                model_obj.save(model_dir)
                saved_version = candidate_version
                break
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"    ⚠ model v{candidate_version} 已存在，嘗試下一個版本...")
                    continue
                raise

        if saved_version is None:
            raise RuntimeError("無法建立新的模型版本，請調整 MODEL_VERSION。")

        print(f"    ✓ 已上傳至 Hopsworks Model Registry：{MODEL_NAME} v{saved_version}")

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
    split_idx = int(len(df_sorted) * 0.8)

    models, tuning_info = train_model(X_train, y_train)
    metrics, y_pred = evaluate_model(
        models, X_test, y_test, feature_names, df_sorted, split_idx
    )
    wf_metrics = evaluate_walk_forward(
        df_sorted,
        feature_names,
        tuning_info["regressor_best_params"],
    )
    metrics.update(wf_metrics)

    mae_pass = metrics["mae"] <= metrics["baseline_mae"] * (1 + MAE_TOLERANCE_PCT)
    rmse_pass = metrics["rmse"] < metrics["baseline_rmse"]
    r2_pass = metrics["r2"] > metrics["baseline_r2"]
    beats_baseline = mae_pass and rmse_pass and r2_pass

    print(f"[5/5] Baseline 守門結果: {'PASS' if beats_baseline else 'FAIL'}")
    print(
        f"    規則: MAE <= baseline*(1+{MAE_TOLERANCE_PCT:.2%}), "
        f"RMSE < baseline, R² > baseline"
    )
    if REGISTER_IF_BEAT_BASELINE and not beats_baseline and not FORCE_REGISTER:
        print("    ⚠ 模型未通過守門條件，本次不上傳新版本。")

    save_model_to_hopsworks(
        models,
        metrics,
        feature_names,
        tuning_info,
        should_register=beats_baseline or (not REGISTER_IF_BEAT_BASELINE),
    )

    print("\n✅ Training Pipeline 執行完畢")
    print(f"   最終 MAPE：{metrics['mape']}%  |  R²：{metrics['r2']}")
    return models, metrics


if __name__ == "__main__":
    models, metrics = main()
