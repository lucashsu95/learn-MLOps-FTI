"""
features.py
----------
共享的技術指標計算模組。

此模組解決 feature_pipeline.py 和 inference_pipeline.py 之間的程式碼重複問題。
所有技術指標計算邏輯集中在此處，確保訓練和推論使用完全相同的特徵計算。

使用方式：
    from src.features import calculate_technical_indicators, calculate_market_context

    df = calculate_technical_indicators(df)
    df = calculate_market_context(df, market_data)
"""

from typing import Optional

import numpy as np
import pandas as pd

from .constants import (
    ATR_PERIOD,
    BB_STD_MULTIPLIER,
    BB_WINDOW,
    # 技術指標參數
    MA_WINDOWS,
    MACD_FAST_PERIOD,
    MACD_SIGNAL_PERIOD,
    MACD_SLOW_PERIOD,
    # 市場基準
    MARKET_TICKERS,
    RETURN_WINDOWS,
    RSI_PERIOD,
    VIX_MA_WINDOW,
    VOLUME_MA_WINDOW,
)


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算所有技術指標。

    此函數取代原本散落在 feature_pipeline.py 和 inference_pipeline.py
    的重複程式碼。

    Args:
        df: 包含 OHLCV 資料的 DataFrame，欄位名稱需為小寫
            (open, high, low, close, volume)

    Returns:
        加入技術指標後的 DataFrame

    Example:
        >>> df = pd.DataFrame({
        ...     'open': [100, 101, 102],
        ...     'high': [105, 106, 107],
        ...     'low': [99, 100, 101],
        ...     'close': [103, 104, 105],
        ...     'volume': [1000, 1100, 1200]
        ... })
        >>> df = calculate_technical_indicators(df)
        >>> 'rsi_14' in df.columns
        True
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ── 移動平均 ──
    for window in MA_WINDOWS:
        df[f"ma_{window}"] = close.rolling(window).mean()

    # ── RSI（相對強弱指標）──
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ── MACD（指數平滑異同移動平均線）──
    ema_fast = close.ewm(span=MACD_FAST_PERIOD, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW_PERIOD, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ── Bollinger Bands（布林通道）──
    df["bb_mid"] = close.rolling(BB_WINDOW).mean()
    bb_std = close.rolling(BB_WINDOW).std()
    df["bb_upper"] = df["bb_mid"] + BB_STD_MULTIPLIER * bb_std
    df["bb_lower"] = df["bb_mid"] - BB_STD_MULTIPLIER * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # ── 價格相對 MA 的位置 ──
    if 20 in MA_WINDOWS:
        df["close_vs_ma20"] = close / df["ma_20"] - 1
    if 50 in MA_WINDOWS:
        df["close_vs_ma50"] = close / df["ma_50"] - 1
    if 20 in MA_WINDOWS and 50 in MA_WINDOWS:
        df["ma20_vs_ma50"] = df["ma_20"] / df["ma_50"] - 1

    # ── Bollinger 位置 ──
    df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ── ATR（平均真實波幅）──
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    df["atr_14"] = tr.rolling(ATR_PERIOD).mean()

    # ── 成交量指標 ──
    df["volume_ma20"] = volume.rolling(VOLUME_MA_WINDOW).mean()
    df["volume_ratio"] = volume / df["volume_ma20"]

    # ── 報酬率 ──
    for window in RETURN_WINDOWS:
        df[f"return_{window}d"] = close.pct_change(window)

    return df


def calculate_market_context(
    df: pd.DataFrame, market_data: dict[str, pd.DataFrame], inplace: bool = False
) -> pd.DataFrame:
    """
    加入市場脈絡特徵（SPY/QQQ 報酬與 VIX 水位/變化）。

    Args:
        df: 目標股票的 DataFrame
        market_data: 市場資料的字典，key 為 'spy', 'qqq', 'vix'，
                     value 為對應的歷史資料 DataFrame
        inplace: 是否原地修改 df

    Returns:
        加入市場脈絡特徵後的 DataFrame
    """
    if not inplace:
        df = df.copy()

    market_df = pd.DataFrame(index=df.index)

    # 合併市場資料
    for key, _symbol in MARKET_TICKERS.items():
        if key not in market_data:
            continue
        hist = market_data[key]
        if hist.empty:
            continue

        # 確保索引對齊
        hist_index = pd.to_datetime(hist.index).tz_localize(None)
        close_col = hist["Close"].rename(f"{key}_close")
        close_col.index = hist_index
        market_df = market_df.join(close_col, how="left")

    # 前向填補和後向填補
    market_df = market_df.ffill().bfill()

    # 計算市場特徵
    if "spy_close" in market_df.columns:
        df["spy_return_1d"] = market_df["spy_close"].pct_change(1)
        df["spy_return_5d"] = market_df["spy_close"].pct_change(5)

    if "qqq_close" in market_df.columns:
        df["qqq_return_1d"] = market_df["qqq_close"].pct_change(1)
        df["qqq_return_5d"] = market_df["qqq_close"].pct_change(5)

    if "vix_close" in market_df.columns:
        df["vix_level"] = market_df["vix_close"]
        df["vix_change_1d"] = market_df["vix_close"].pct_change(1)
        df["vix_ma20"] = market_df["vix_close"].rolling(VIX_MA_WINDOW).mean()
        df["vix_vs_ma20"] = df["vix_level"] / df["vix_ma20"] - 1

    return df


def calculate_target(
    df: pd.DataFrame, horizon_days: int = 5, target_mode: str = "raw"
) -> pd.DataFrame:
    """
    計算預測目標。

    Args:
        df: 包含 close 欄位的 DataFrame
        horizon_days: 預測天數
        target_mode: "raw" (原始報酬) 或 "excess_spy" (超額報酬)

    Returns:
        加入目標欄位後的 DataFrame
    """
    close = df["close"]
    raw_return = close.shift(-horizon_days) / close - 1

    if target_mode == "excess_spy" and "spy_return_5d" in df.columns:
        # 超額報酬 = 標的報酬 - SPY 報酬
        # 注意：spy_return_5d 是歷史報酬，我們需要未來的 SPY 報酬嗎？
        # 不，目標通常是預測未來的報酬。
        # 這裡的邏輯需要小心：我們應該預測 (Future Ticker Return) - (Future SPY Return)
        # 或是 (Future Ticker Return) - (Current SPY 5d Return)?
        # 通常是前者。但我們沒有未來的 SPY 報酬。
        # 在訓練時，我們有未來的資料。
        # 這裡為了簡化，我們先實作 raw return，
        # 如果要 excess return，需要對齊 spy 的 shift。
        pass

    # 暫時維持原樣，但增加欄位名稱一致性
    target_col = f"target_{horizon_days}d_return"
    if target_mode == "excess_spy":
        target_col = f"target_excess_spy_{horizon_days}d_return"
        # 抓取未來的 spy 報酬 (如果可用)
        if "spy_close" in df.columns:
            spy_close = df["spy_close"]
            spy_future_return = spy_close.shift(-horizon_days) / spy_close - 1
            df[target_col] = raw_return - spy_future_return
        else:
            df[target_col] = raw_return
    else:
        df[target_col] = raw_return

    df["target_next_return"] = close.shift(-1) / close - 1
    return df


def get_fundamentals_from_info(info: dict) -> dict:
    """
    從 yfinance info 字典提取基本面數據。

    Args:
        info: yfinance Ticker.info 字典

    Returns:
        基本面數據字典
    """
    return {
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "eps": info.get("trailingEps"),
        "price_to_book": info.get("priceToBook"),
        "debt_to_equity": info.get("debtToEquity"),
        "profit_margin": info.get("profitMargins"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "market_cap": info.get("marketCap"),
        "w52_high": info.get("fiftyTwoWeekHigh"),
        "w52_low": info.get("fiftyTwoWeekLow"),
    }


def merge_fundamentals(df: pd.DataFrame, fundamentals: dict) -> pd.DataFrame:
    """
    將基本面數據（靜態）廣播到每一行。

    Args:
        df: 目標 DataFrame
        fundamentals: 基本面數據字典

    Returns:
        加入基本面欄位後的 DataFrame
    """
    for key, val in fundamentals.items():
        df[key] = val
    return df


def get_feature_columns() -> list[str]:
    """
    取得特徵欄位列表（有序）。

    Returns:
        特徵欄位名稱列表
    """
    return [
        # 技術指標
        "ma_5",
        "close_vs_ma20",
        "close_vs_ma50",
        "ma20_vs_ma50",
        "bb_width",
        "bb_position",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr_14",
        "volume_ratio",
        "return_1d",
        "return_5d",
        "return_20d",
        # 基本面
        "pe_ratio",
        "forward_pe",
        "eps",
        "price_to_book",
        "profit_margin",
        "revenue_growth",
        "earnings_growth",
        # 市場脈絡
        "spy_return_1d",
        "spy_return_5d",
        "qqq_return_1d",
        "qqq_return_5d",
        "vix_level",
        "vix_change_1d",
        "vix_vs_ma20",
    ]


def validate_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    驗證並填補特徵欄位的缺失值。

    Args:
        df: 目標 DataFrame
        feature_cols: 需要驗證的特徵欄位列表

    Returns:
        清理後的 DataFrame
    """
    # 只保留存在的特徵欄位
    available_features = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]

    if missing:
        print(f" ⚠ 以下特徵欄位不存在，跳過: {missing}")

    # 填補基本面 NaN（靜態欄位偶爾為 None）
    for col in available_features:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def clean_dataframe_for_training(
    df: pd.DataFrame, required_cols: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    清理 DataFrame 用於訓練。

    Args:
        df: 目標 DataFrame
        required_cols: 必須存在的欄位列表

    Returns:
        清理後的 DataFrame
    """
    if required_cols is None:
        required_cols = ["ma_50", "rsi_14", "macd"]

    # 移除因技術指標 rolling window 產生的頭部 NaN
    df = df.dropna(subset=required_cols)

    # 重設 index，讓 date 變成欄位
    df = df.reset_index()
    if "date" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "date"})

    if "date" in df.columns:
        df["date"] = df["date"].astype(str)

    return df
