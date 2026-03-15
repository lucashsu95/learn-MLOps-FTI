"""
抓取股票歷史數據、計算技術指標、加入財報基本面數據，
最後寫入 Hopsworks Feature Store。

使用方式：
    pip install yfinance pandas_ta hopsworks scikit-learn
    python feature_pipeline.py

環境變數（或直接改 HOPSWORKS_API_KEY 常數）：
    export HOPSWORKS_API_KEY="your_api_key_here"
"""

import os
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ── 設定區（改這裡就好） ──────────────────────────────────────────
TICKER = os.environ.get("TICKER", "AAPL").upper()  # 要預測的股票代號
PERIOD = "3y"            # 歷史資料長度（2y / 3y / 5y）
HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT")   # Hopsworks 上的 project 名稱
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
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
# ─────────────────────────────────────────────────────────────────


def fetch_price_data(ticker: str, period: str) -> pd.DataFrame:
    """下載 OHLCV 歷史數據"""
    print(f"[1/4] 下載 {ticker} 歷史價格資料（{period}）...")
    t = yf.Ticker(ticker)
    df = t.history(period=period)

    if df.empty:
        raise ValueError(f"找不到 {ticker} 的資料，請確認代號是否正確。")

    # 整理欄位
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)  # 移除 timezone
    df.index.name = "date"
    df.columns = [c.lower() for c in df.columns]
    print(f"    ✓ 取得 {len(df)} 筆資料，從 {df.index[0].date()} 到 {df.index[-1].date()}")
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算技術指標（不依賴 pandas_ta，手動計算確保穩定）"""
    print("[2/4] 計算技術指標...")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ── 移動平均 ──
    df["ma_5"]  = close.rolling(5).mean()
    df["ma_20"] = close.rolling(20).mean()
    df["ma_50"] = close.rolling(50).mean()

    # ── RSI（14日）──
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ── MACD ──
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── Bollinger Bands ──
    df["bb_mid"]   = close.rolling(20).mean()
    bb_std         = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    df["close_vs_ma20"] = close / df["ma_20"] - 1
    df["close_vs_ma50"] = close / df["ma_50"] - 1
    df["ma20_vs_ma50"]  = df["ma_20"] / df["ma_50"] - 1
    df["bb_position"]   = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ── ATR（平均真實波幅）──
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # ── 成交量指標 ──
    df["volume_ma20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["volume_ma20"]  # 今日量 / 20日均量

    # ── 價格動能 ──
    df["return_1d"]  = close.pct_change(1)
    df["return_5d"]  = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)

    # ── 目標欄位：明日報酬率 ──
    df["target_next_return"] = close.shift(-1) / close - 1

    print(f"    ✓ 新增 {len([c for c in df.columns if c not in ['open','high','low','close','volume']])} 個特徵欄位")
    return df


def fetch_fundamental_data(ticker: str) -> dict:
    """抓取基本面數據（財報資料，每季更新一次）"""
    print("[3/4] 抓取基本面數據...")
    t = yf.Ticker(ticker)
    info = t.info

    fundamentals = {
        "pe_ratio":         info.get("trailingPE",       None),
        "forward_pe":       info.get("forwardPE",        None),
        "eps":              info.get("trailingEps",      None),
        "price_to_book":    info.get("priceToBook",      None),
        "debt_to_equity":   info.get("debtToEquity",     None),
        "profit_margin":    info.get("profitMargins",    None),
        "revenue_growth":   info.get("revenueGrowth",    None),
        "earnings_growth":  info.get("earningsGrowth",   None),
        "market_cap":       info.get("marketCap",        None),
        "w52_high":         info.get("fiftyTwoWeekHigh", None),
        "w52_low":          info.get("fiftyTwoWeekLow",  None),
    }

    # 印出哪些拿到、哪些是 None
    missing = [k for k, v in fundamentals.items() if v is None]
    if missing:
        print(f"    ⚠ 以下基本面欄位為 None（yfinance 未回傳）: {missing}")
    available = [k for k, v in fundamentals.items() if v is not None]
    print(f"    ✓ 成功取得: {available}")

    return fundamentals


def merge_fundamentals(df: pd.DataFrame, fundamentals: dict) -> pd.DataFrame:
    """將基本面數據（靜態）廣播到每一行"""
    for key, val in fundamentals.items():
        df[key] = val
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """清理：移除 NaN、重設 index、型別轉換"""
    # 移除因技術指標 rolling window 產生的頭部 NaN
    df = df.dropna(subset=["ma_50", "rsi_14", "macd", "target_next_return"])

    # 最後一筆 target 是 NaN（沒有明日），移除
    df = df.dropna(subset=["target_next_return"])

    # 重設 index，讓 date 變成欄位
    df = df.reset_index()
    df["date"] = df["date"].astype(str)

    # 加入 ticker 欄位
    df.insert(0, "ticker", TICKER)

    print(f"    ✓ 清理後剩 {len(df)} 筆，欄位數: {len(df.columns)}")
    return df


def save_to_hopsworks(df: pd.DataFrame) -> None:
    """寫入 Hopsworks Feature Store"""
    print("[4/4] 連接 Hopsworks 並寫入 Feature Store...")
    try:
        import hopsworks
    except ImportError:
        print("    ✗ hopsworks 套件未安裝，執行: pip install hopsworks")
        print("    （本地 debug 模式：印出前 5 筆資料）")
        print(df.head())
        return

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY
    )
    fs = project.get_feature_store()

    used_version = None

    # 先從設定版本開始，若遇到「table already exists」衝突，往上遞增版本。
    for offset in range(5):
        candidate_version = FEATURE_GROUP_VERSION + offset
        try:
            fg = fs.get_or_create_feature_group(
                name=FEATURE_GROUP_NAME,
                version=candidate_version,
                primary_key=["ticker", "date"],
                description=f"{TICKER} stock features: technical indicators and fundamentals",
                online_enabled=False,
            )
            fg.insert(df, write_options={"wait_for_job": True})
            used_version = candidate_version
            break
        except Exception as e:
            err_msg = str(e)

            # 常見情況：metadata 建立/寫入時，Hive table 已存在。
            if "already exists" in err_msg.lower() and "table" in err_msg.lower():
                print(f"    ⚠ v{candidate_version} 建立失敗（table 已存在），嘗試下一個版本...")
                continue

            # 既有版本 schema 不相容時，升版建立新 schema。
            if "not compatible with feature group schema" in err_msg.lower() or "does not exist in feature group" in err_msg.lower():
                print(f"    ⚠ v{candidate_version} schema 不相容，嘗試下一個版本...")
                continue
            raise

    if used_version is None:
        raise RuntimeError("無法建立或取得可用的 Feature Group，請檢查 Hopsworks 專案中的既有資料表。")

    print(f"    ✓ 成功寫入 {len(df)} 筆資料到 Feature Group: {FEATURE_GROUP_NAME} v{used_version}")


def main():
    print("=" * 55)
    print(f"  Stock Feature Pipeline  |  Ticker: {TICKER}")
    print("=" * 55)

    df = fetch_price_data(TICKER, PERIOD)
    df = add_technical_indicators(df)

    fundamentals = fetch_fundamental_data(TICKER)
    df = merge_fundamentals(df, fundamentals)
    df = clean_dataframe(df)

    print(f"\n最終 DataFrame 欄位列表：")
    for col in df.columns:
        print(f"  - {col}")

    save_to_hopsworks(df)

    print("\n✅ Feature Pipeline 執行完畢")
    return df  # 方便 notebook 中直接使用


if __name__ == "__main__":
    result_df = main()
