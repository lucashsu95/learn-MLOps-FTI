"""
constants.py
-----------
所有魔術數字與常數的集中定義。

修改任何參數只需改這裡，不用翻遍所有檔案。
"""

# ─────────────────────────────────────────────────────────────────
# 技術指標參數
# ─────────────────────────────────────────────────────────────────

# 移動平均線
MA_WINDOWS = [5, 20, 50]

# RSI
RSI_PERIOD = 14

# MACD
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

# Bollinger Bands
BB_WINDOW = 20
BB_STD_MULTIPLIER = 2.0

# ATR（平均真實波幅）
ATR_PERIOD = 14

# 成交量
VOLUME_MA_WINDOW = 20

# 報酬率計算
RETURN_WINDOWS = [1, 5, 20]


# ─────────────────────────────────────────────────────────────────
# 市場基準
# ─────────────────────────────────────────────────────────────────

MARKET_TICKERS = {
    "spy": "SPY",
    "qqq": "QQQ",
    "vix": "^VIX",
}

VIX_MA_WINDOW = 20


# ─────────────────────────────────────────────────────────────────
# 資料相關
# ─────────────────────────────────────────────────────────────────

# yfinance 資料期間
DEFAULT_HISTORY_PERIOD = "3y"
INFERENCE_HISTORY_PERIOD = "120d"  # 推論時需要足夠長度計算 MA50 / RSI14
UI_HISTORY_DAYS = 90

# Fallback 期間列表（從長到短）
FALLBACK_PERIODS = ["3y", "5y", "3y", "2y", "1y"]


# ─────────────────────────────────────────────────────────────────
# 模型訓練參數
# ─────────────────────────────────────────────────────────────────

# 時序切分
TRAIN_TEST_SPLIT_RATIO = 0.8

# 時序交叉驗證
TIME_SERIES_SPLITS = 5
WALK_FORWARD_SPLITS_DEFAULT = 5

# RandomizedSearchCV
RANDOM_SEARCH_ITERATIONS = 18
RANDOM_SEARCH_RANDOM_STATE = 42

# 樣本權重（近期加權）
DEFAULT_HALF_LIFE_DAYS = 252.0

# Baseline 守門
DEFAULT_MAE_TOLERANCE_PCT = 0.01


# ─────────────────────────────────────────────────────────────────
# XGBoost 參數
# ─────────────────────────────────────────────────────────────────

XGB_RANDOM_STATE = 42
XGB_N_JOBS = -1

# 分類器預設參數
XGB_CLASSIFIER_DEFAULTS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

# RandomizedSearchCV 超參數範圍
XGB_PARAM_DIST = {
    "n_estimators": [200, 300, 500, 800, 1200],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 8, 10],
    "reg_alpha": [0.0, 0.1, 0.3, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}


# ─────────────────────────────────────────────────────────────────
# 訊號與門檻
# ─────────────────────────────────────────────────────────────────

DEFAULT_SIGNAL_THRESHOLD = 0.58
PREDICTION_CLIP_RANGE = (-0.2, 0.2)  # 預測值裁切範圍


# ─────────────────────────────────────────────────────────────────
# Feature Store / Model Registry
# ─────────────────────────────────────────────────────────────────

FEATURE_GROUP_VERSION_MAX_OFFSET = 100
MODEL_VERSION_MAX_OFFSET = 20
FEATURE_GROUP_MAX_VERSION_SEARCH = 150


# ─────────────────────────────────────────────────────────────────
# Modal 排程
# ─────────────────────────────────────────────────────────────────

# 推論排程：每週一至五 UTC 22:00（台灣時間隔日 06:00）
INFERENCE_CRON_SCHEDULE = "0 22 * * 1-5"
INFERENCE_TIMEOUT = 300  # 秒

# 重訓排程：每週六 UTC 01:30（台灣時間週六 09:30）
RETRAINING_CRON_SCHEDULE = "30 1 * * 6"
RETRAINING_TIMEOUT = 1800  # 秒

# Modal 映像檔套件
MODAL_INFERENCE_PACKAGES = [
    "yfinance",
    "hopsworks",
    "xgboost",
    "scikit-learn",
    "pandas",
    "numpy",
    "joblib",
]

MODAL_RETRAINING_PACKAGES = [
    "hopsworks",
    "xgboost",
    "scikit-learn",
    "matplotlib",
    "pandas",
    "numpy",
    "python-dotenv",
    "yfinance",
]


# ─────────────────────────────────────────────────────────────────
# UI 相關
# ─────────────────────────────────────────────────────────────────

GRADIO_SERVER_PORT = 7860
GRADIO_SERVER_NAME = "0.0.0.0"


# ─────────────────────────────────────────────────────────────────
# 預設值
# ─────────────────────────────────────────────────────────────────

DEFAULT_TICKER = "AAPL"
DEFAULT_TARGET_HORIZON_DAYS = 5
DEFAULT_TARGET_MODE = "excess_spy"
DEFAULT_FEATURE_GROUP_VERSION = 1
DEFAULT_MODEL_VERSION = 1
DEFAULT_PREDICTION_GROUP_VERSION = 1
