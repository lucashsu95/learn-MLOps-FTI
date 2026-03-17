"""
conftest.py
----------
pytest fixtures 和測試配置。
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# 確保可以 import src 模組
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─────────────────────────────────────────────────────────────────
# 測試資料 Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """
    產生測試用的 OHLCV 資料。

    Returns:
        包含 100 筆 OHLCV 資料的 DataFrame
    """
    np.random.seed(42)
    n = 100

    # 產生合理的股價資料
    base_price = 150.0
    returns = np.random.randn(n) * 0.02  # 2% 標準差的日報酬
    prices = base_price * np.exp(np.cumsum(returns))

    # OHLC 資料
    high = prices * (1 + np.abs(np.random.randn(n)) * 0.01)
    low = prices * (1 - np.abs(np.random.randn(n)) * 0.01)
    open_prices = prices + np.random.randn(n) * 0.5
    close = prices
    volume = np.random.randint(1000000, 10000000, n)

    dates = pd.date_range(
        start="2023-01-01",
        periods=n,
        freq="B"  # 工作日
    )

    df = pd.DataFrame({
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    df.index.name = "date"
    return df


@pytest.fixture
def sample_market_data() -> dict:
    """
    產生測試用的市場資料（SPY, QQQ, VIX）。

    Returns:
        包含市場資料的字典
    """
    np.random.seed(42)
    n = 100

    dates = pd.date_range(start="2023-01-01", periods=n, freq="B")

    # SPY
    spy_close = 400 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    spy = pd.DataFrame({"Close": spy_close}, index=dates)

    # QQQ
    qqq_close = 350 * np.exp(np.cumsum(np.random.randn(n) * 0.012))
    qqq = pd.DataFrame({"Close": qqq_close}, index=dates)

    # VIX
    vix_close = 15 + np.abs(np.random.randn(n) * 5)
    vix = pd.DataFrame({"Close": vix_close}, index=dates)

    return {
        "spy": spy,
        "qqq": qqq,
        "vix": vix,
    }


@pytest.fixture
def sample_fundamentals() -> dict:
    """
    產生測試用的基本面資料。

    Returns:
        基本面資料字典
    """
    return {
        "pe_ratio": 25.5,
        "forward_pe": 23.2,
        "eps": 5.88,
        "price_to_book": 45.0,
        "debt_to_equity": 1.5,
        "profit_margin": 0.25,
        "revenue_growth": 0.08,
        "earnings_growth": 0.12,
        "market_cap": 2500000000000,
        "w52_high": 199.62,
        "w52_low": 124.17,
    }


# ─────────────────────────────────────────────────────────────────
# 環境 Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """
    設定測試環境變數。

    自動套用到所有測試，確保環境變數存在。
    """
    monkeypatch.setenv("TICKER", "AAPL")
    monkeypatch.setenv("HOPSWORKS_PROJECT", "test_project")
    monkeypatch.setenv("HOPSWORKS_API_KEY", "test_api_key")
    monkeypatch.setenv("FEATURE_GROUP_VERSION", "1")
    monkeypatch.setenv("MODEL_VERSION", "1")
    monkeypatch.setenv("TARGET_HORIZON_DAYS", "5")
    monkeypatch.setenv("TARGET_MODE", "excess_spy")
    monkeypatch.setenv("SIGNAL_THRESHOLD", "0.58")


@pytest.fixture
def mock_hopsworks(monkeypatch):
    """
    Mock Hopsworks 連線。

    用於不需要真實 Hopsworks 連線的測試。
    """

    class MockFeatureStore:
        def get_feature_group(self, _name, _version):
            return None

        def get_or_create_feature_group(self, **_kwargs):
            class MockFG:
                def insert(self, _df, **_kwargs):
                    pass

            return MockFG()

    class MockProject:
        def get_feature_store(self):
            return MockFeatureStore()

        def get_model_registry(self):
            return MockModelRegistry()

    class MockModelRegistry:
        def get_model(self, _name, _version):
            return None

        def sklearn(self):
            class MockSklearn:
                @staticmethod
                def create_model(**_kwargs):
                    class MockModel:
                        @staticmethod
                        def save(_path):
                            pass

                    return MockModel()

            return MockSklearn()

    def mock_login(*_args, **_kwargs):
        return MockProject()

    # Monkey patch hopsworks.login
    import hopsworks

    monkeypatch.setattr(hopsworks, "login", mock_login)


# ─────────────────────────────────────────────────────────────────
# 輔助函數
# ─────────────────────────────────────────────────────────────────

def assert_valid_technical_indicators(df: pd.DataFrame):
    """
    驗證技術指標計算結果是否合理。

    Args:
        df: 包含技術指標的 DataFrame

    Raises:
        AssertionError: 如果技術指標超出合理範圍
    """
    # RSI 應該在 0-100 之間
    if "rsi_14" in df.columns:
        rsi_valid = df["rsi_14"].dropna()
        assert rsi_valid.min() >= 0, f"RSI 最小值 {rsi_valid.min()} 小於 0"
        assert rsi_valid.max() <= 100, f"RSI 最大值 {rsi_valid.max()} 大於 100"

    # Bollinger 上軌應該大於中軌，中軌應該大於下軌
    if all(col in df.columns for col in ["bb_upper", "bb_mid", "bb_lower"]):
        valid_rows = df.dropna(subset=["bb_upper", "bb_mid", "bb_lower"])
        assert (valid_rows["bb_upper"] >= valid_rows["bb_mid"]).all(), "BB 上軌小於中軌"
        assert (valid_rows["bb_mid"] >= valid_rows["bb_lower"]).all(), "BB 中軌小於下軌"

    # MA 應該是正值
    for col in ["ma_5", "ma_20", "ma_50"]:
        if col in df.columns:
            ma_valid = df[col].dropna()
            assert (ma_valid > 0).all(), f"{col} 包含非正值"


def create_test_config():
    """
    建立測試用的 Config 物件。

    Returns:
        Config 實例
    """
    from src.config import Config

    return Config(
        ticker="AAPL",
        period="1y",
        hopsworks_project="test_project",
        hopsworks_api_key="test_api_key",
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
