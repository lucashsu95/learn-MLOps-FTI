"""
retraining_pipeline.py
----------------------
自動重訓入口：可用 Modal 排程週期性重訓。

使用方式：
    modal deploy retraining_pipeline.py
    modal run retraining_pipeline.py

預設排程：每週六 UTC 01:30（台灣時間週六 09:30）。
"""

import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

MODAL_IMAGE_PACKAGES = [
    "hopsworks",
    "xgboost",
    "scikit-learn",
    "matplotlib",
    "pandas",
    "numpy",
    "python-dotenv",
    "yfinance",
]


def run_retraining_once():
    """執行一次重訓主流程。"""
    from training_pipeline import main as training_main

    print("=" * 55)
    print(f"  Scheduled Retraining  |  {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 55)
    training_main()


try:
    import modal

    image = modal.Image.debian_slim().pip_install(*MODAL_IMAGE_PACKAGES)
    app = modal.App("stock-retraining-pipeline")

    secret = modal.Secret.from_dict(
        {
            "HOPSWORKS_API_KEY": os.environ.get("HOPSWORKS_API_KEY", ""),
            "HOPSWORKS_PROJECT": os.environ.get("HOPSWORKS_PROJECT", ""),
            "TICKER": os.environ.get("TICKER", "AAPL"),
            "FEATURE_GROUP_VERSION": os.environ.get("FEATURE_GROUP_VERSION", "1"),
            "MODEL_VERSION": os.environ.get("MODEL_VERSION", "1"),
            "TARGET_HORIZON_DAYS": os.environ.get("TARGET_HORIZON_DAYS", "5"),
            "TARGET_MODE": os.environ.get("TARGET_MODE", "excess_spy"),
            "SIGNAL_THRESHOLD": os.environ.get("SIGNAL_THRESHOLD", "0.58"),
            "REGISTER_IF_BEAT_BASELINE": os.environ.get("REGISTER_IF_BEAT_BASELINE", "1"),
            "FORCE_REGISTER": os.environ.get("FORCE_REGISTER", "0"),
            "WALK_FORWARD_SPLITS": os.environ.get("WALK_FORWARD_SPLITS", "5"),
            "HALF_LIFE_DAYS": os.environ.get("HALF_LIFE_DAYS", "252"),
            "MAE_TOLERANCE_PCT": os.environ.get("MAE_TOLERANCE_PCT", "0.01"),
        }
    )

    @app.function(
        image=image,
        secrets=[secret],
        schedule=modal.Cron("30 1 * * 6"),
        timeout=1800,
    )
    def scheduled_retraining():
        """Modal 每週自動重訓。"""
        run_retraining_once()

    @app.local_entrypoint()
    def main():
        """modal run retraining_pipeline.py 時執行。"""
        run_retraining_once()

except ImportError:
    if __name__ == "__main__":
        run_retraining_once()
