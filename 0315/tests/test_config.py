"""
test_config.py
-----------
測試配置管理模組。
"""

import os

import pytest

from src.config import (
    FEATURE_COLS,
    Config,
    _get_bool_env,
    _get_float_env,
    _get_int_env,
    _validate_required,
    get_config,
    get_target_col,
)


class TestConfigDataclass:
    """測試 Config dataclass。"""

    def test_config_properties(self):
        """測試 Config 屬性計算。"""
        config = Config(
            ticker="AAPL",
            period="3y",
            hopsworks_project="test_project",
            hopsworks_api_key="test_key",
            feature_group_name="aapl_stock_features",
            feature_group_version=1,
            model_name="aapl_xgb_regressor",
            model_version=1,
            target_horizon_days=5,
            target_mode="excess_spy",
            signal_threshold=0.58,
            half_life_days=252.0,
            mae_tolerance_pct=0.01,
            walk_forward_splits=5,
            register_if_beat_baseline=True,
            force_register=False,
            prediction_group_name="aapl_predictions",
            prediction_group_version=1,
        )

        assert config.feature_group_full_name == "aapl_stock_features"
        assert config.model_full_name == "aapl_xgb_regressor"
        assert config.prediction_group_full_name == "aapl_predictions"

    def test_config_uppercase_ticker(self):
        """ticker 應該轉為大寫。"""
        config = Config(
            ticker="aapl",  # 小寫輸入
            period="3y",
            hopsworks_project="test_project",
            hopsworks_api_key="test_key",
            feature_group_name="aapl_stock_features",
            feature_group_version=1,
            model_name="aapl_xgb_regressor",
            model_version=1,
            target_horizon_days=5,
            target_mode="excess_spy",
            signal_threshold=0.58,
            half_life_days=252.0,
            mae_tolerance_pct=0.01,
            walk_forward_splits=5,
            register_if_beat_baseline=True,
            force_register=False,
            prediction_group_name="aapl_predictions",
            prediction_group_version=1,
        )

        # 注意：Config 不會自動轉大寫，這是在 get_config() 中處理的
        assert config.ticker == "aapl"


class TestGetConfig:
    """測試 get_config 函數。"""

    def test_get_config_from_env(self, setup_test_env):
        """測試從環境變數載入配置。"""
        config = get_config()

        assert config.ticker == "AAPL"
        assert config.hopsworks_project == "test_project"
        assert config.target_horizon_days == 5
        assert config.target_mode == "excess_spy"

    def test_get_config_ticker_override(self, setup_test_env):
        """測試 ticker 覆蓋。"""
        config = get_config(ticker="TSLA")

        assert config.ticker == "TSLA"
        assert config.feature_group_name == "tsla_stock_features"
        assert config.model_name == "tsla_xgb_regressor"

    def test_get_config_period_override(self, setup_test_env):
        """測試 period 覆蓋。"""
        config = get_config(period="5y")

        assert config.period == "5y"

    def test_get_config_uppercase_ticker(self, setup_test_env):
        """測試 ticker 自動轉大寫。"""
        config = get_config(ticker="msft")

        assert config.ticker == "MSFT"

    def test_get_config_invalid_target_mode(self, setup_test_env):
        """測試無效的 target_mode。"""
        # 需要直接設置環境變數
        original = os.environ.get("TARGET_MODE")
        os.environ["TARGET_MODE"] = "invalid_mode"

        try:
            with pytest.raises(ValueError, match="TARGET_MODE 僅支援 raw / excess_spy"):
                get_config()
        finally:
            if original is not None:
                os.environ["TARGET_MODE"] = original
            else:
                os.environ.pop("TARGET_MODE", None)


class TestEnvHelpers:
    """測試環境變數輔助函數。"""

    def test_get_int_env_with_value(self):
        """測試 _get_int_env 正常值。"""
        os.environ["TEST_INT"] = "42"
        result = _get_int_env("TEST_INT", 1)
        assert result == 42

    def test_get_int_env_with_default(self):
        """測試 _get_int_env 使用預設值。"""
        os.environ.pop("TEST_INT_MISSING", None)
        result = _get_int_env("TEST_INT_MISSING", 10)
        assert result == 10

    def test_get_int_env_with_empty_string(self):
        """測試 _get_int_env 空字串。"""
        os.environ["TEST_INT_EMPTY"] = ""
        result = _get_int_env("TEST_INT_EMPTY", 10)
        assert result == 10

    def test_get_int_env_with_invalid_value(self):
        """測試 _get_int_env 無效值。"""
        os.environ["TEST_INT_INVALID"] = "not_a_number"

        with pytest.raises(ValueError, match="必須是整數"):
            _get_int_env("TEST_INT_INVALID", 10)

    def test_get_float_env_with_value(self):
        """測試 _get_float_env 正常值。"""
        os.environ["TEST_FLOAT"] = "3.14"
        result = _get_float_env("TEST_FLOAT", 1.0)
        assert result == pytest.approx(3.14)

    def test_get_float_env_with_default(self):
        """測試 _get_float_env 使用預設值。"""
        os.environ.pop("TEST_FLOAT_MISSING", None)
        result = _get_float_env("TEST_FLOAT_MISSING", 2.5)
        assert result == 2.5

    def test_get_bool_env_true(self):
        """測試 _get_bool_env True 值。"""
        os.environ["TEST_BOOL"] = "1"
        assert _get_bool_env("TEST_BOOL", False) is True

    def test_get_bool_env_false(self):
        """測試 _get_bool_env False 值。"""
        os.environ["TEST_BOOL"] = "0"
        assert _get_bool_env("TEST_BOOL", True) is False

    def test_get_bool_env_default(self):
        """測試 _get_bool_env 使用預設值。"""
        os.environ.pop("TEST_BOOL_MISSING", None)
        assert _get_bool_env("TEST_BOOL_MISSING", True) is True

    def test_validate_required_with_value(self):
        """測試 _validate_required 正常值。"""
        os.environ["TEST_REQUIRED"] = "value"
        result = _validate_required("TEST_REQUIRED", os.environ.get("TEST_REQUIRED"))
        assert result == "value"

    def test_validate_required_missing(self):
        """測試 _validate_required 缺失值。"""
        with pytest.raises(ValueError, match="缺少 TEST_MISSING"):
            _validate_required("TEST_MISSING", None)


class TestGetTargetCol:
    """測試目標欄位名稱計算。"""

    def test_get_target_col_raw(self):
        """測試 raw 模式的目標欄位名稱。"""
        result = get_target_col("raw", 5)
        assert result == "target_5d_return"

    def test_get_target_col_excess_spy(self):
        """測試 excess_spy 模式的目標欄位名稱。"""
        result = get_target_col("excess_spy", 5)
        assert result == "target_excess_spy_5d_return"

    def test_get_target_col_different_horizon(self):
        """測試不同預測天期的目標欄位名稱。"""
        result = get_target_col("raw", 10)
        assert result == "target_10d_return"


class TestFeatureCols:
    """測試特徵欄位列表。"""

    def test_feature_cols_not_empty(self):
        """特徵欄位列表不應該是空的。"""
        assert len(FEATURE_COLS) > 0

    def test_feature_cols_contains_technical(self):
        """特徵欄位應該包含技術指標。"""
        technical_features = [
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "close_vs_ma20", "close_vs_ma50",
            "bb_width", "bb_position",
            "atr_14", "volume_ratio",
        ]

        for feat in technical_features:
            assert feat in FEATURE_COLS, f"缺少技術指標: {feat}"

    def test_feature_cols_contains_fundamental(self):
        """特徵欄位應該包含基本面。"""
        fundamental_features = [
            "pe_ratio", "forward_pe", "eps",
            "price_to_book", "profit_margin",
            "revenue_growth", "earnings_growth",
        ]

        for feat in fundamental_features:
            assert feat in FEATURE_COLS, f"缺少基本面特徵: {feat}"

    def test_feature_cols_contains_market(self):
        """特徵欄位應該包含市場脈絡。"""
        market_features = [
            "spy_return_1d", "spy_return_5d",
            "qqq_return_1d", "qqq_return_5d",
            "vix_level", "vix_change_1d", "vix_vs_ma20",
        ]

        for feat in market_features:
            assert feat in FEATURE_COLS, f"缺少市場特徵: {feat}"
