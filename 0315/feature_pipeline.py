"""
feature_pipeline.py
------------------
抓取股票歷史數據、計算技術指標、加入財報基本面數據，
最後寫入 Hopsworks Feature Store。

使用方式：
    pip install yfinance pandas hopsworks
    python feature_pipeline.py

環境變數（或直接改 .env）：
    TICKER=AAPL
    HOPSWORKS_PROJECT=your_project_name
    HOPSWORKS_API_KEY=your_api_key
"""

import os
import sys
import warnings

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# 匯入共享模組
sys.path.insert(0, os.path.dirname(__file__))
from src.constants import (
    DEFAULT_HISTORY_PERIOD,
    FEATURE_GROUP_VERSION_MAX_OFFSET,
    MARKET_TICKERS,
)
from src.features import (
    calculate_market_context,
    calculate_target,
    calculate_technical_indicators,
    get_fundamentals_from_info,
    merge_fundamentals,
)
from src.utils import (
    print_error,
    print_section,
    print_step,
    print_success,
    print_warning,
)

warnings.filterwarnings("ignore")
load_dotenv()

# ── 設定區（向後相容）────────────────────────────────────────────
TICKER = os.environ.get("TICKER", "AAPL").upper()
PERIOD = os.environ.get("PERIOD", DEFAULT_HISTORY_PERIOD)
HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
FEATURE_GROUP_NAME = f"{TICKER.lower()}_stock_features"


def _get_int_env(name: str, default: int) -> int:
    """從環境變數取得整數值（向後相容）。"""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} 必須是整數，收到: {raw}") from e


FEATURE_GROUP_VERSION = _get_int_env("FEATURE_GROUP_VERSION", 1)
TARGET_HORIZON_DAYS = _get_int_env("TARGET_HORIZON_DAYS", 5)
TARGET_MODE = os.environ.get("TARGET_MODE", "excess_spy").strip().lower()
# ─────────────────────────────────────────────────────────────────


def _safe_history(symbol: str, period: str) -> pd.DataFrame:
    """下載行情並在失敗時回退較短期間。"""
    fallback_periods = [period, "5y", "3y", "2y", "1y"]
    seen = set()
    for p in fallback_periods:
        if p in seen:
            continue
        seen.add(p)
        try:
            hist = yf.Ticker(symbol).history(period=p)
            if hist is not None and not hist.empty:
                if p != period:
                    print_warning(f"{symbol} 期間 {period} 失敗，改用 {p}")
                return hist
        except Exception:
            continue
    return pd.DataFrame()


def fetch_price_data(ticker: str, period: str) -> pd.DataFrame:
    """下載 OHLCV 歷史數據"""
    print_step(1, 4, f"下載 {ticker} 歷史價格資料（{period}）...")
    df = _safe_history(ticker, period)

    if df.empty:
        raise ValueError(f"找不到 {ticker} 的資料，請確認代號是否正確。")

    # 整理欄位
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df.columns = [c.lower() for c in df.columns]

    # 這裡使用 .date() 或 .strftime() 取得日期字串，避免 LSP 報錯
    start_date = df.index[0].strftime("%Y-%m-%d")
    end_date = df.index[-1].strftime("%Y-%m-%d")
    print_success(f"取得 {len(df)} 筆資料，從 {start_date} 到 {end_date}")
    return df  # type: ignore


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算技術指標（使用共享模組）"""
    print_step(2, 4, "計算技術指標...")
    df = calculate_technical_indicators(df)

    # 計算新增的特徵數量
    original_cols = {"open", "high", "low", "close", "volume"}
    new_features = [c for c in df.columns if c not in original_cols]
    print_success(f"新增 {len(new_features)} 個特徵欄位")
    return df


def add_market_features(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """加入市場脈絡特徵：SPY/QQQ 報酬與 VIX 水位/變化。"""
    print_step(2.5, 4, "加入市場脈絡特徵（SPY / QQQ / VIX）...")

    market_data = {}
    for key, symbol in MARKET_TICKERS.items():
        hist = _safe_history(symbol, period)
        if hist.empty:
            print_warning(f"{symbol} 無資料，略過 {key} 特徵")
            continue
        market_data[key] = hist

    df = calculate_market_context(df, market_data)

    # 報告新增的市場特徵
    market_cols = [c for c in df.columns if any(k in c for k in ["spy", "qqq", "vix"])]
    print_success(f"新增市場特徵: {market_cols}")
    return df


def fetch_fundamental_data(ticker: str) -> dict:
    """抓取基本面數據（財報資料，每季更新一次）"""
    print_step(3, 4, "抓取基本面數據...")
    t = yf.Ticker(ticker)
    info = t.info

    fundamentals = get_fundamentals_from_info(info)

    # 印出哪些拿到、哪些是 None
    missing = [k for k, v in fundamentals.items() if v is None]
    if missing:
        print_warning(f"以下基本面欄位為 None（yfinance 未回傳）: {missing}")
    available = [k for k, v in fundamentals.items() if v is not None]
    print_success(f"成功取得: {available}")

    return fundamentals


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """清理：移除 NaN、重設 index、型別轉換"""
    from src.config import get_target_col

    target_col = get_target_col(TARGET_MODE, TARGET_HORIZON_DAYS)

    # 移除因技術指標 rolling window 產生的頭部 NaN
    required_cols = ["ma_50", "rsi_14", "macd", "target_next_return", target_col]
    market_required = [
        "spy_return_1d",
        "spy_return_5d",
        "qqq_return_1d",
        "qqq_return_5d",
        "vix_level",
        "vix_change_1d",
        "vix_vs_ma20",
    ]
    required_cols.extend([c for c in market_required if c in df.columns])
    df = df.dropna(subset=required_cols)

    # 最後一筆 target 是 NaN（沒有明日），移除
    df = df.dropna(subset=["target_next_return", target_col])

    # 重設 index，讓 date 變成欄位
    df = df.reset_index()
    df["date"] = df["date"].astype(str)

    # 加入 ticker 欄位
    df.insert(0, "ticker", TICKER)

    print_success(f"清理後剩 {len(df)} 筆，欄位數: {len(df.columns)}")
    return df


def save_to_hopsworks(df: pd.DataFrame) -> None:
    """寫入 Hopsworks Feature Store"""
    print_step(4, 4, "連接 Hopsworks 並寫入 Feature Store...")
    try:
        import hopsworks
    except ImportError:
        print_error("hopsworks 套件未安裝，執行: pip install hopsworks")
        print_warning("（本地 debug 模式：印出前 5 筆資料）")
        print(df.head())
        return

    project = hopsworks.login(project=HOPSWORKS_PROJECT, api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    used_version = None

    # 先從設定版本開始，若遇到「table already exists」衝突，往上遞增版本。
    for offset in range(FEATURE_GROUP_VERSION_MAX_OFFSET):
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
                print_warning(f"v{candidate_version} 建立失敗（table 已存在），嘗試下一個版本...")
                continue

            # 既有版本 schema 不相容時，升版建立新 schema。
            schema_errors = [
                "not compatible with feature group schema",
                "does not exist in feature group",
                "cannot insert data into feature group",
                "field",
                "schema",
            ]
            if any(msg in err_msg.lower() for msg in schema_errors):
                print_warning(f"v{candidate_version} schema 不相容或寫入失敗，嘗試下一個版本...")
                print_warning(f"錯誤細節: {err_msg[:200]}...")
                continue

            # 若不是已知可跳過的錯誤，則拋出
            print_error(f"v{candidate_version} 發生未知錯誤：{err_msg}")
            raise

    if used_version is None:
        raise RuntimeError(
            "無法建立或取得可用的 Feature Group，請檢查 Hopsworks 專案中的既有資料表。"
        )

    print_success(
        f"成功寫入 {len(df)} 筆資料到 Feature Group: {FEATURE_GROUP_NAME} v{used_version}"
    )


def main():
    print_section(f" Stock Feature Pipeline | Ticker: {TICKER}")

    df = fetch_price_data(TICKER, PERIOD)
    df = add_technical_indicators(df)
    df = add_market_features(df, PERIOD)

    fundamentals = fetch_fundamental_data(TICKER)
    df = merge_fundamentals(df, fundamentals)

    # 使用共享模組計算目標，傳入正確參數
    df = calculate_target(df, horizon_days=TARGET_HORIZON_DAYS, target_mode=TARGET_MODE)

    df = clean_dataframe(df)

    print("\n最終 DataFrame 欄位列表：")
    for col in df.columns:
        print(f" - {col}")

    save_to_hopsworks(df)

    print("\n✅ Feature Pipeline 執行完畢")
    return df  # 方便 notebook 中直接使用


if __name__ == "__main__":
    result_df = main()
