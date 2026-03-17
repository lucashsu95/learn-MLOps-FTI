"""
test_features.py
--------------
測試技術指標計算模組。
"""

import numpy as np
import pandas as pd

from src.constants import (
    MA_WINDOWS,
)
from src.features import (
    calculate_market_context,
    calculate_target,
    calculate_technical_indicators,
    get_feature_columns,
    get_fundamentals_from_info,
    merge_fundamentals,
    validate_features,
)


class TestCalculateTechnicalIndicators:
    """測試技術指標計算函數。"""

    def test_basic_calculation(self, sample_ohlcv_data):
        """測試基本技術指標計算。"""
        df = calculate_technical_indicators(sample_ohlcv_data)

        # 檢查是否產生了所有預期的欄位
        expected_cols = [
            "ma_5", "ma_20", "ma_50",
            "rsi_14",
            "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_lower", "bb_mid", "bb_width",
            "atr_14",
            "volume_ratio",
            "return_1d", "return_5d", "return_20d",
        ]

        for col in expected_cols:
            assert col in df.columns, f"缺少欄位: {col}"

    def test_rsi_range(self, sample_ohlcv_data):
        """RSI 應該在 0-100 之間。"""
        df = calculate_technical_indicators(sample_ohlcv_data)

        rsi_valid = df["rsi_14"].dropna()
        assert rsi_valid.min() >= 0, f"RSI 最小值 {rsi_valid.min()} 小於 0"
        assert rsi_valid.max() <= 100, f"RSI 最大值 {rsi_valid.max()} 大於 100"

    def test_moving_averages(self, sample_ohlcv_data):
        """移動平均線應該是正值且合理。"""
        df = calculate_technical_indicators(sample_ohlcv_data)

        for window in MA_WINDOWS:
            col = f"ma_{window}"
            ma_valid = df[col].dropna()

            # MA 應該是正值
            assert (ma_valid > 0).all(), f"{col} 包含非正值"

            # MA 不應該偏離收盤價太多（誤差容忍 20%）
            close_valid = df["close"].loc[ma_valid.index]
            ratio = ma_valid / close_valid
            assert (ratio > 0.8).all() and (ratio < 1.2).all(), \
                f"{col} 偏離收盤價太多"

    def test_bollinger_bands(self, sample_ohlcv_data):
        """Bollinger Bands 的上軌應該大於中軌，中軌應該大於下軌。"""
        df = calculate_technical_indicators(sample_ohlcv_data)

        valid_rows = df.dropna(subset=["bb_upper", "bb_mid", "bb_lower"])

        assert (valid_rows["bb_upper"] >= valid_rows["bb_mid"]).all(), \
            "BB 上軌小於中軌"
        assert (valid_rows["bb_mid"] >= valid_rows["bb_lower"]).all(), \
            "BB 中軌小於下軌"

    def test_macd_calculation(self, sample_ohlcv_data):
        """MACD 計算應該正確。"""
        df = calculate_technical_indicators(sample_ohlcv_data)

        valid_rows = df.dropna(subset=["macd", "macd_signal", "macd_hist"])

        # MACD histogram = MACD - Signal
        calculated_hist = valid_rows["macd"] - valid_rows["macd_signal"]
        pd.testing.assert_series_equal(
            valid_rows["macd_hist"],
            calculated_hist,
            check_names=False,
            rtol=1e-10
        )

    def test_returns_calculation(self, sample_ohlcv_data):
        """報酬率計算應該正確。"""
        df = calculate_technical_indicators(sample_ohlcv_data)

        # 手動計算 1 日報酬率
        manual_return_1d = df["close"].pct_change(1)

        pd.testing.assert_series_equal(
            df["return_1d"].dropna(),
            manual_return_1d.dropna(),
            check_names=False,
            rtol=1e-10
        )

    def test_atr_positive(self, sample_ohlcv_data):
        """ATR 應該是正值。"""
        df = calculate_technical_indicators(sample_ohlcv_data)

        atr_valid = df["atr_14"].dropna()
        assert (atr_valid > 0).all(), "ATR 包含非正值"

    def test_volume_ratio(self, sample_ohlcv_data):
        """成交量比率應該是正值。"""
        df = calculate_technical_indicators(sample_ohlcv_data)

        vol_ratio_valid = df["volume_ratio"].dropna()
        assert (vol_ratio_valid > 0).all(), "成交量比率包含非正值"


class TestCalculateMarketContext:
    """測試市場脈絡特徵計算。"""

    def test_market_features_added(self, sample_ohlcv_data, sample_market_data):
        """測試市場特徵是否正確加入。"""
        df = calculate_market_context(sample_ohlcv_data, sample_market_data)

        expected_cols = [
            "spy_return_1d", "spy_return_5d",
            "qqq_return_1d", "qqq_return_5d",
            "vix_level", "vix_change_1d", "vix_vs_ma20",
        ]

        for col in expected_cols:
            assert col in df.columns, f"缺少市場特徵: {col}"

    def test_vix_level_range(self, sample_ohlcv_data, sample_market_data):
        """VIX 水準應該是正值。"""
        df = calculate_market_context(sample_ohlcv_data, sample_market_data)

        vix_valid = df["vix_level"].dropna()
        assert (vix_valid > 0).all(), "VIX 包含非正值"

    def test_not_inplace(self, sample_ohlcv_data, sample_market_data):
        """測試預設不原地修改。"""
        original_cols = list(sample_ohlcv_data.columns)
        _ = calculate_market_context(sample_ohlcv_data, sample_market_data)

        # 原始 DataFrame 不應該被修改
        assert list(sample_ohlcv_data.columns) == original_cols


class TestCalculateTarget:
    """測試目標欄位計算。"""

    def test_target_calculation(self, sample_ohlcv_data):
        """測試目標欄位計算。"""
        df = calculate_target(sample_ohlcv_data, horizon_days=5)

        assert "target_5d_return" in df.columns
        assert "target_next_return" in df.columns

    def test_target_values(self, sample_ohlcv_data):
        """目標值應該合理。"""
        df = calculate_target(sample_ohlcv_data, horizon_days=5)

        # 排除最後幾個 NaN（因為 shift 產生）
        valid = df["target_5d_return"].dropna()

        # 日報酬率通常在 ±20% 範圍內
        assert (valid > -0.3).all() and (valid < 0.3).all(), \
            f"目標報酬率超出合理範圍: min={valid.min():.2%}, max={valid.max():.2%}"


class TestFundamentals:
    """測試基本面資料處理。"""

    def test_get_fundamentals_from_info(self):
        """測試從 yfinance info 提取基本面。"""
        mock_info = {
            "trailingPE": 25.5,
            "forwardPE": 23.2,
            "trailingEps": 5.88,
            "priceToBook": 45.0,
            "debtToEquity": 1.5,
            "profitMargins": 0.25,
            "revenueGrowth": 0.08,
            "earningsGrowth": 0.12,
            "marketCap": 2500000000000,
            "fiftyTwoWeekHigh": 199.62,
            "fiftyTwoWeekLow": 124.17,
        }

        result = get_fundamentals_from_info(mock_info)

        assert result["pe_ratio"] == 25.5
        assert result["eps"] == 5.88
        assert result["market_cap"] == 2500000000000

    def test_merge_fundamentals(self, sample_ohlcv_data, sample_fundamentals):
        """測試基本面資料合併。"""
        df = merge_fundamentals(sample_ohlcv_data, sample_fundamentals)

        for key in sample_fundamentals:
            assert key in df.columns, f"缺少基本面欄位: {key}"
            # 所有行的值應該相同（廣播）
            assert (df[key] == sample_fundamentals[key]).all(), \
                f"{key} 的值不一致"


class TestGetFeatureColumns:
    """測試特徵欄位列表。"""

    def test_feature_columns_list(self):
        """特徵欄位列表應該包含所有必要欄位。"""
        cols = get_feature_columns()

        # 應該是列表
        assert isinstance(cols, list)

        # 應該包含技術指標
        assert "rsi_14" in cols
        assert "macd" in cols
        assert "close_vs_ma20" in cols

        # 應該包含基本面
        assert "pe_ratio" in cols
        assert "eps" in cols

        # 應該包含市場脈絡
        assert "spy_return_1d" in cols
        assert "vix_level" in cols


class TestValidateFeatures:
    """測試特徵驗證。"""

    def test_validate_features_removes_missing(self, sample_ohlcv_data):
        """驗證應該只保留存在的特徵。"""
        df = calculate_technical_indicators(sample_ohlcv_data)

        # 只驗證存在的欄位
        feature_cols = ["rsi_14", "macd", "nonexistent_col"]
        result = validate_features(df, feature_cols)

        # nonexistent_col 應該被跳過
        assert "nonexistent_col" not in result.columns

    def test_validate_features_fills_nan(self, sample_ohlcv_data):
        """驗證應該填補 NaN。"""
        df = calculate_technical_indicators(sample_ohlcv_data)

        # 加入一些 NaN（使用 .iloc 處理 DatetimeIndex）
        rsi_idx = df.columns.get_loc("rsi_14")
        df.iloc[0:6, rsi_idx] = np.nan

        result = validate_features(df, ["rsi_14"])

        # NaN 應該被填補
        assert not result["rsi_14"].isna().any()
