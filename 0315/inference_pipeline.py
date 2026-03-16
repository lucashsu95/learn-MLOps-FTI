"""
每個交易日收盤後自動執行：
  1. 抓取今日最新價格 & 技術指標
  2. 從 Hopsworks Model Registry 載入模型
  3. 預測明日收盤價
  4. 將預測結果寫回 Hopsworks Feature Store（供 UI 讀取）

部署方式：
    pip install modal hopsworks yfinance numpy pandas
    modal deploy inference_pipeline.py     # 部署到 Modal（長期排程）
    modal run   inference_pipeline.py      # 手動觸發一次（測試用）

本地測試（不需要 Modal）：
    python inference_pipeline.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
import hopsworks
import joblib

warnings.filterwarnings("ignore")
load_dotenv()

# ── 設定區 ────────────────────────────────────────────────────────
TICKER = os.environ.get("TICKER", "AAPL").upper()
HOPSWORKS_PROJECT  = os.environ.get("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY  = os.environ.get("HOPSWORKS_API_KEY")


def _get_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} 必須是整數，收到: {raw}") from e

FEATURE_GROUP_NAME      = f"{TICKER.lower()}_stock_features"
FEATURE_GROUP_VERSION   = _get_int_env("FEATURE_GROUP_VERSION", 1)
PREDICTION_GROUP_NAME   = f"{TICKER.lower()}_predictions"
PREDICTION_GROUP_VERSION = _get_int_env("PREDICTION_GROUP_VERSION", 1)
MODEL_NAME    = f"{TICKER.lower()}_xgb_regressor"
MODEL_VERSION = _get_int_env("MODEL_VERSION", 1)
SIGNAL_THRESHOLD = float(os.environ.get("SIGNAL_THRESHOLD", "0.58"))

# Modal Image（部署時的 Python 環境）
MODAL_IMAGE_PACKAGES = [
    "yfinance", "hopsworks", "xgboost",
    "scikit-learn", "pandas", "numpy", "joblib"
]
# ─────────────────────────────────────────────────────────────────

# Import from shared modules
from src.features import calculate_technical_indicators, calculate_market_context, get_fundamentals_from_info
from src.config import get_config
from src.constants import MARKET_TICKERS, INFERENCE_HISTORY_PERIOD, PREDICTION_CLIP_RANGE
from src.utils import add_trading_days, print_success, print_warning, print_step, print_section


def _validate_hopsworks_config() -> None:
    """確保必要環境變數存在，避免用假值誤連線。"""
    if not HOPSWORKS_PROJECT:
        raise ValueError("缺少 HOPSWORKS_PROJECT，請在 .env 或系統環境變數設定。")
    if not HOPSWORKS_API_KEY:
        raise ValueError("缺少 HOPSWORKS_API_KEY，請在 .env 或系統環境變數設定。")


def _add_trading_days(base_date: date, days: int) -> date:
    """從 base_date 起往後推算 N 個交易日（僅跳過週末）。"""
    return add_trading_days(base_date, days)


# ── Modal 設定區 ──────────────────────────────────────────────────
def _get_modal_app():
    """動態載入 Modal，避免本地執行時報錯"""
    import modal

    _validate_hopsworks_config()

    image = modal.Image.debian_slim().pip_install(*MODAL_IMAGE_PACKAGES)
    app   = modal.App("stock-inference-pipeline")

    # Hopsworks API Key 用 Modal Secret 存（安全）
    secret = modal.Secret.from_dict({"HOPSWORKS_API_KEY": HOPSWORKS_API_KEY})

    return app, image, secret


# ── Step 1：取得最新特徵 ───────────────────────────────────────────
def fetch_latest_features(ticker: str) -> tuple:
    """抓今日最新一筆資料，計算出所有特徵，回傳一個 Series"""
    import yfinance as yf
    
    print_step(1, 4, f"抓取 {ticker} 最新資料...")
    
    t = yf.Ticker(ticker)
    df = t.history(period=INFERENCE_HISTORY_PERIOD)
    
    if df.empty:
        raise ValueError(f"無法取得 {ticker} 資料")
    
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.columns = [c.lower() for c in df.columns]
    
    # 使用共享模組計算技術指標（取代原本 60+ 行重複程式碼）
    df = calculate_technical_indicators(df)
    
    # 市場脈絡
    market_data = {}
    for key, symbol in MARKET_TICKERS.items():
        hist = yf.Ticker(symbol).history(period=INFERENCE_HISTORY_PERIOD)
        if not hist.empty:
            market_data[key] = hist
    
    df = calculate_market_context(df, market_data)
    
    # 基本面
    info = t.info
    fundamentals = get_fundamentals_from_info(info)
    for key, val in fundamentals.items():
        df[key] = val
    
    latest = df.iloc[-1]
    latest_date = df.index[-1].date()
    print_success(f"最新資料日期：{latest_date} 收盤價：{latest['close']:.2f}")
    
    return latest, latest_date


# ── Step 2：載入模型 ──────────────────────────────────────────────
def load_model_from_hopsworks():
    """從 Hopsworks Model Registry 載入模型與 metadata"""
    print("  [2/4] 從 Model Registry 載入模型...")
    import hopsworks
    import joblib

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=os.environ.get("HOPSWORKS_API_KEY", HOPSWORKS_API_KEY)
    )
    mr = project.get_model_registry()
    model_obj = mr.get_model(MODEL_NAME, version=MODEL_VERSION)
    model_dir = model_obj.download()

    reg_path = os.path.join(model_dir, "reg_model.pkl")
    cls_path = os.path.join(model_dir, "cls_model.pkl")
    legacy_path = os.path.join(model_dir, "model.pkl")

    reg_model = joblib.load(reg_path if os.path.exists(reg_path) else legacy_path)
    cls_model = joblib.load(cls_path) if os.path.exists(cls_path) else None

    with open(os.path.join(model_dir, "metadata.json")) as f:
        metadata = json.load(f)

    feature_cols = metadata["feature_cols"]
    print(f"  ✓ 模型載入成功（訓練於 {metadata['trained_at'][:10]}）")
    print(f"  ✓ 使用 {len(feature_cols)} 個特徵")
    return {
        "regressor": reg_model,
        "classifier": cls_model,
    }, feature_cols, metadata


def load_model_from_local(model_dir: str = None):
    """本地 fallback：從 model_aapl/ 資料夾讀取"""
    import joblib
    if model_dir is None:
        model_dir = f"model_{TICKER.lower()}"
    reg_path = os.path.join(model_dir, "reg_model.pkl")
    cls_path = os.path.join(model_dir, "cls_model.pkl")
    legacy_path = os.path.join(model_dir, "model.pkl")
    reg_model = joblib.load(reg_path if os.path.exists(reg_path) else legacy_path)
    cls_model = joblib.load(cls_path) if os.path.exists(cls_path) else None
    with open(os.path.join(model_dir, "metadata.json")) as f:
        metadata = json.load(f)
    print(f"  ✓ 本地模型載入：{model_dir}/")
    return {
        "regressor": reg_model,
        "classifier": cls_model,
    }, metadata["feature_cols"], metadata


# ── Step 3：執行預測 ──────────────────────────────────────────────
def run_inference(models, feature_cols: list, latest_row: pd.Series, latest_date: date, metadata: dict = None) -> dict:
    """用最新特徵預測目標期報酬率，並換算預測收盤價"""
    print("  [3/4] 執行推論...")

    # 對齊特徵欄位（缺失的填 median 或 0）
    X = {}
    for col in feature_cols:
        val = latest_row.get(col, None)
        X[col] = float(val) if (val is not None and not pd.isna(val)) else 0.0

    X_df = pd.DataFrame([X])
    reg_model = models["regressor"]
    cls_model = models.get("classifier")

    reg_pred = float(reg_model.predict(X_df)[0])
    reg_pred = float(np.clip(reg_pred, -0.2, 0.2))

    used_threshold = float((metadata or {}).get("tuning", {}).get("signal_threshold", SIGNAL_THRESHOLD))
    if cls_model is not None:
        up_prob = float(cls_model.predict_proba(X_df)[0][1])
        if up_prob >= used_threshold:
            model_output = abs(reg_pred)
        elif up_prob <= (1 - used_threshold):
            model_output = -abs(reg_pred)
        else:
            model_output = 0.0
    else:
        up_prob = None
        model_output = reg_pred

    target_mode = str((metadata or {}).get("target_mode", "raw")).lower()
    horizon_days = int((metadata or {}).get("target_horizon_days", 1))
    if target_mode == "excess_spy":
        benchmark_col = f"spy_return_{horizon_days}d"
        benchmark_return = float(latest_row.get(benchmark_col, 0.0) or 0.0)
        predicted_return = model_output + benchmark_return
    else:
        benchmark_return = 0.0
        predicted_return = model_output

    predicted_return = float(np.clip(predicted_return, -0.2, 0.2))

    current_price   = float(latest_row["close"])
    predicted_price = current_price * (1 + predicted_return)
    change_pct      = predicted_return * 100
    direction       = "⬆ 看漲" if change_pct > 0 else "⬇ 看跌"
    predicted_for_date = _add_trading_days(latest_date, horizon_days)

    result = {
        "ticker":          TICKER,
        "prediction_date": str(latest_date),
        "predicted_for_date": str(predicted_for_date),
        "target_horizon_days": horizon_days,
        "current_close":   round(current_price, 4),
        "predicted_close": round(predicted_price, 4),
        "change_pct":      round(change_pct, 4),
        "direction":       direction,
        "predicted_at":    datetime.now().isoformat(),
    }

    print(f"  ✓ 今日收盤：${current_price:.2f}")
    print(f"  ✓ 預測 {horizon_days} 日後：${predicted_price:.2f}  ({change_pct:+.2f}%)  {direction}")
    if target_mode == "excess_spy":
        print(f"  ✓ 目標模式：超額報酬（加回 SPY {horizon_days}d 報酬 {benchmark_return:+.4f}）")
    if up_prob is not None:
        print(f"  ✓ Up 機率：{up_prob:.4f}  |  信心門檻：{used_threshold:.2f}")
    return result


# ── Step 4：寫回 Hopsworks ────────────────────────────────────────
def save_prediction_to_hopsworks(result: dict):
    """將預測結果寫入 predictions feature group（供 UI 讀取）"""
    print("  [4/4] 寫入預測結果至 Hopsworks...")
    try:
        import hopsworks

        project = hopsworks.login(
            project=HOPSWORKS_PROJECT,
            api_key_value=os.environ.get("HOPSWORKS_API_KEY", HOPSWORKS_API_KEY)
        )
        fs = project.get_feature_store()

        df_pred = pd.DataFrame([result])

        used_version = None
        for offset in range(5):
            candidate_version = PREDICTION_GROUP_VERSION + offset
            try:
                fg = fs.get_or_create_feature_group(
                    name=PREDICTION_GROUP_NAME,
                    version=candidate_version,
                    primary_key=["ticker", "prediction_date"],
                    description=f"{TICKER} daily close prediction results",
                    online_enabled=True,   # 開啟 online store 供 UI 即時讀取
                )
                fg.insert(df_pred, write_options={"wait_for_job": True})
                used_version = candidate_version
                break
            except Exception as e:
                err_msg = str(e)
                if "already exists" in err_msg.lower() and "table" in err_msg.lower():
                    print(f"  ⚠ v{candidate_version} 建立失敗（table 已存在），嘗試下一個版本...")
                    continue
                if "not compatible with feature group schema" in err_msg.lower() or "does not exist in feature group" in err_msg.lower():
                    print(f"  ⚠ v{candidate_version} schema 不相容，嘗試下一個版本...")
                    continue
                raise

        if used_version is None:
            raise RuntimeError("無法建立或取得可用的 predictions Feature Group。")

        print(f"  ✓ 預測結果已寫入：{PREDICTION_GROUP_NAME} v{used_version}")

    except Exception as e:
        print(f"  ⚠ Hopsworks 寫入失敗（本地模式）: {e}")
        # 本地 fallback：存成 JSON
        out_path = "latest_prediction.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 預測結果已存至本地：{out_path}")


# ── 核心執行函式（Modal & 本地共用）────────────────────────────────
def run_pipeline():
    print_section(f"  Stock Inference Pipeline  |  {datetime.now():%Y-%m-%d %H:%M}")

    _validate_hopsworks_config()

    # Step 1
    latest_row, latest_date = fetch_latest_features(TICKER)

    # Step 2：優先嘗試 Hopsworks，失敗用本地
    try:
        models, feature_cols, metadata = load_model_from_hopsworks()
    except Exception as e:
        print(f"  ⚠ Hopsworks 連線失敗，改用本地模型: {e}")
        models, feature_cols, metadata = load_model_from_local()

    # Step 3
    result = run_inference(models, feature_cols, latest_row, latest_date, metadata)

    # Step 4
    save_prediction_to_hopsworks(result)

    print("\n✅ Inference Pipeline 執行完畢")
    horizon_days = int((metadata or {}).get("target_horizon_days", 1))
    print(f"   {result['ticker']}  預測 {horizon_days} 日後收盤：${result['predicted_close']}  {result['direction']}")
    return result


# ── Modal 排程部署 ────────────────────────────────────────────────
try:
    import modal

    app, image, secret = _get_modal_app()

    @app.function(
        image=image,
        secrets=[secret],
        schedule=modal.Cron("0 22 * * 1-5"),  # 每週一至五 UTC 22:00（台灣時間隔日 06:00）
        timeout=300,
    )
    def scheduled_inference():
        """Modal 自動排程：每個交易日執行"""
        return run_pipeline()

    @app.local_entrypoint()
    def main():
        """modal run inference_pipeline.py 時執行"""
        run_pipeline()

except ImportError:
    pass


if __name__ == "__main__":
    run_pipeline()
