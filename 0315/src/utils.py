"""
utils.py
--------
日誌與工具函數。

取代散落各處的 print()，提供統一的日誌介面。
"""

import logging
import sys
from datetime import date, timedelta

# ─────────────────────────────────────────────────────────────────
# 日誌設定
# ─────────────────────────────────────────────────────────────────

# 建立專案專用的 logger
logger = logging.getLogger("mlops_fti")

# 預設日誌格式
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    format_str: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> None:
    """
    設定專案日誌。

    Args:
        level: 日誌等級 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_str: 日誌格式字串
        date_format: 日期格式字串
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # 移除既有的 handlers（避免重複）
    logger.handlers.clear()

    # 設定 handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(format_str, date_format))

    logger.addHandler(handler)
    logger.setLevel(log_level)

    # 不向上傳播
    logger.propagate = False


# ─────────────────────────────────────────────────────────────────
# 輸出輔助函數（相容現有 print 風格）
# ─────────────────────────────────────────────────────────────────

def log_success(message: str) -> None:
    """記錄成功訊息。"""
    logger.info(f"✓ {message}")


def log_warning(message: str) -> None:
    """記錄警告訊息。"""
    logger.warning(f"⚠ {message}")


def log_error(message: str) -> None:
    """記錄錯誤訊息。"""
    logger.error(f"✗ {message}")


def log_step(step: int, total: int, message: str) -> None:
    """記錄步驟訊息。"""
    logger.info(f"[{step}/{total}] {message}")


def log_section(title: str) -> None:
    """記錄區段標題。"""
    logger.info("=" * 55)
    logger.info(title)
    logger.info("=" * 55)


def log_alert(message: str) -> None:
    """記錄告警訊息（集中告警輸出）。"""
    logger.warning(f"[ALERT] {message}")


# ─────────────────────────────────────────────────────────────────
# 向後相容的 print 替代（用於漸進遷移）
# ─────────────────────────────────────────────────────────────────

def print_success(message: str) -> None:
    """列印成功訊息（向後相容）。"""
    print(f" ✓ {message}")


def print_warning(message: str) -> None:
    """列印警告訊息（向後相容）。"""
    print(f" ⚠ {message}")


def print_error(message: str) -> None:
    """列印錯誤訊息（向後相容）。"""
    print(f" ✗ {message}")


def print_step(step: int, total: int, message: str) -> None:
    """列印步驟訊息（向後相容）。"""
    print(f"[{step}/{total}] {message}")


def print_section(title: str) -> None:
    """列印區段標題（向後相容）。"""
    print("=" * 55)
    print(title)
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────
# 檔案操作輔助
# ─────────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """確保目錄存在。"""
    import os
    os.makedirs(path, exist_ok=True)


def read_json(filepath: str) -> dict:
    """讀取 JSON 檔案。"""
    import json
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def write_json(filepath: str, data: dict) -> None:
    """寫入 JSON 檔案。"""
    import json
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────
# 日期工具
# ─────────────────────────────────────────────────────────────────


def add_trading_days(base_date: date, days: int) -> date:
    """
    從 base_date 起往後推算 N 個交易日（僅跳過週末）。

    Args:
        base_date: 基準日期
        days: 要增加的交易日數

    Returns:
        計算後的日期
    """
    d = base_date
    remaining = max(int(days), 0)
    while remaining > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:  # 0-4 為週一到週五
            remaining -= 1
    return d


def next_trading_day(date_str: str) -> str:
    """
    計算下一個交易日（簡單跳過週末）。

    Args:
        date_str: 日期字串 (YYYY-MM-DD)

    Returns:
        下一個交易日字串
    """
    from datetime import datetime

    d = datetime.strptime(date_str[:10], "%Y-%m-%d") + timedelta(days=1)
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d += timedelta(days=1)
    return d.strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────
# 初始化
# ─────────────────────────────────────────────────────────────────

# 預設設定基本日誌（如果尚未設定）
if not logger.handlers:
    setup_logging()
