"""
config.py
---------
統一的環境變數管理與配置。

使用方式：
    from src.config import get_config

    config = get_config()
    print(config.ticker)
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# 載入 .env
load_dotenv()


@dataclass
class Config:
    """專案配置的單一來源。"""

    # ── 基本設定 ──
    ticker: str
    period: str

    # ── Hopsworks 設定 ──
    hopsworks_project: str
    hopsworks_api_key: str

    # ── Feature Store 設定 ──
    feature_group_name: str
    feature_group_version: int

    # ── Model Registry 設定 ──
    model_name: str
    model_version: int

    # ── 預測目標設定 ──
    target_horizon_days: int
    target_mode: str  # "raw" or "excess_spy"

    # ── 訓練設定 ──
    signal_threshold: float
    half_life_days: float
    mae_tolerance_pct: float
    walk_forward_splits: int

    # ── 開關 ──
    register_if_beat_baseline: bool
    force_register: bool

    # ── 推論設定 ──
    prediction_group_name: str
    prediction_group_version: int

    @property
    def feature_group_full_name(self) -> str:
        """Feature Group 完整名稱。"""
        return f"{self.ticker.lower()}_stock_features"

    @property
    def model_full_name(self) -> str:
        """Model 完整名稱。"""
        return f"{self.ticker.lower()}_xgb_regressor"

    @property
    def prediction_group_full_name(self) -> str:
        """Prediction Group 完整名稱。"""
        return f"{self.ticker.lower()}_predictions"


def _get_int_env(name: str, default: int) -> int:
    """從環境變數取得整數值。"""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} 必須是整數，收到: {raw}") from e


def _get_float_env(name: str, default: float) -> float:
    """從環境變數取得浮點數值。"""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{name} 必須是數字，收到: {raw}") from e


def _get_bool_env(name: str, default: bool) -> bool:
    """從環境變數取得布林值。"""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw == "1"


def _validate_required(name: str, value: Optional[str]) -> str:
    """驗證必要環境變數存在。"""
    if not value:
        raise ValueError(f"缺少 {name}，請在 .env 或系統環境變數設定。")
    return value


def get_config(
    ticker: Optional[str] = None,
    period: Optional[str] = None,
) -> Config:
    """
    從環境變數載入配置。

    Args:
        ticker: 可選的股票代號覆蓋
        period: 可選的歷史期間覆蓋

    Returns:
        Config 實例

    Raises:
        ValueError: 如果必要環境變數缺失或格式錯誤
    """
    # 基本設定
    raw_ticker = ticker or os.environ.get("TICKER", "AAPL")
    config_ticker = raw_ticker.upper()
    config_period = period or os.environ.get("PERIOD", "3y")

    # Hopsworks 設定（必要）
    hopsworks_project = _validate_required(
        "HOPSWORKS_PROJECT",
        os.environ.get("HOPSWORKS_PROJECT")
    )
    hopsworks_api_key = _validate_required(
        "HOPSWORKS_API_KEY",
        os.environ.get("HOPSWORKS_API_KEY")
    )

    # Feature Store 設定
    feature_group_version = _get_int_env("FEATURE_GROUP_VERSION", 1)
    feature_group_name = f"{config_ticker.lower()}_stock_features"

    # Model Registry 設定
    model_version = _get_int_env("MODEL_VERSION", 1)
    model_name = f"{config_ticker.lower()}_xgb_regressor"

    # 預測目標設定
    target_horizon_days = _get_int_env("TARGET_HORIZON_DAYS", 5)
    target_mode = os.environ.get("TARGET_MODE", "excess_spy").strip().lower()

    if target_mode not in {"raw", "excess_spy"}:
        raise ValueError(f"TARGET_MODE 僅支援 raw / excess_spy，收到: {target_mode}")

    # 訓練設定
    signal_threshold = _get_float_env("SIGNAL_THRESHOLD", 0.58)
    half_life_days = _get_float_env("HALF_LIFE_DAYS", 252.0)
    mae_tolerance_pct = _get_float_env("MAE_TOLERANCE_PCT", 0.01)
    walk_forward_splits = _get_int_env("WALK_FORWARD_SPLITS", 5)

    # 開關
    register_if_beat_baseline = _get_bool_env("REGISTER_IF_BEAT_BASELINE", True)
    force_register = _get_bool_env("FORCE_REGISTER", False)

    # 推論設定
    prediction_group_version = _get_int_env("PREDICTION_GROUP_VERSION", 1)
    prediction_group_name = f"{config_ticker.lower()}_predictions"

    return Config(
        ticker=config_ticker,
        period=config_period,
        hopsworks_project=hopsworks_project,
        hopsworks_api_key=hopsworks_api_key,
        feature_group_name=feature_group_name,
        feature_group_version=feature_group_version,
        model_name=model_name,
        model_version=model_version,
        target_horizon_days=target_horizon_days,
        target_mode=target_mode,
        signal_threshold=signal_threshold,
        half_life_days=half_life_days,
        mae_tolerance_pct=mae_tolerance_pct,
        walk_forward_splits=walk_forward_splits,
        register_if_beat_baseline=register_if_beat_baseline,
        force_register=force_register,
        prediction_group_name=prediction_group_name,
        prediction_group_version=prediction_group_version,
    )


# ─────────────────────────────────────────────────────────────────
# 特徵欄位定義（與 feature_pipeline.py 對應）
# ─────────────────────────────────────────────────────────────────

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


def get_target_col(target_mode: str, target_horizon_days: int) -> str:
    """取得目標欄位名稱。"""
    if target_mode == "raw":
        return f"target_{target_horizon_days}d_return"
    else:
        return f"target_excess_spy_{target_horizon_days}d_return"
