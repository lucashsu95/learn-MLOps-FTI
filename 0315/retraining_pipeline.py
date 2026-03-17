"""
retraining_pipeline.py
----------------------
自動重訓入口：可用 Modal 排程週期性重訓。

使用方式：
    modal deploy retraining_pipeline.py
    modal run retraining_pipeline.py

預設排程：每週六 UTC 01:30（台灣時間週六 09:30）。
"""

from datetime import datetime

from src.config import get_config
from src.constants import MODAL_RETRAINING_PACKAGES, RETRAINING_CRON_SCHEDULE, RETRAINING_TIMEOUT

# 取得配置
config = get_config()


def run_retraining_once():
    """執行一次重訓主流程。"""
    from training_pipeline import main as training_main

    print("=" * 55)
    print(f"  Scheduled Retraining  |  {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 55)
    training_main()


try:
    import modal

    image = modal.Image.debian_slim().pip_install(*MODAL_RETRAINING_PACKAGES)
    app = modal.App("stock-retraining-pipeline")

    secret = modal.Secret.from_dict(
        {
            "HOPSWORKS_API_KEY": config.hopsworks_api_key,
            "HOPSWORKS_PROJECT": config.hopsworks_project,
            "TICKER": config.ticker,
            "FEATURE_GROUP_VERSION": str(config.feature_group_version),
            "MODEL_VERSION": str(config.model_version),
            "TARGET_HORIZON_DAYS": str(config.target_horizon_days),
            "TARGET_MODE": config.target_mode,
            "SIGNAL_THRESHOLD": str(config.signal_threshold),
            "REGISTER_IF_BEAT_BASELINE": "1" if config.register_if_beat_baseline else "0",
            "FORCE_REGISTER": "1" if config.force_register else "0",
            "WALK_FORWARD_SPLITS": str(config.walk_forward_splits),
            "HALF_LIFE_DAYS": str(config.half_life_days),
            "MAE_TOLERANCE_PCT": str(config.mae_tolerance_pct),
        }
    )

    @app.function(
        image=image,
        secrets=[secret],
        schedule=modal.Cron(RETRAINING_CRON_SCHEDULE),
        timeout=RETRAINING_TIMEOUT,
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
