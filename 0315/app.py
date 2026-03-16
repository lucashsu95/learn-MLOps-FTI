"""
app.py
------
Hugging Face Spaces 上的 Gradio UI。
從 Hopsworks 讀取最新預測 + 歷史走勢，以互動圖表呈現。

部署方式：
1. 在 HF Spaces 新建 Space（SDK: Gradio）
2. 上傳此檔案 + requirements.txt
3. 在 Space Settings → Secrets 加入 HOPSWORKS_PROJECT 與 HOPSWORKS_API_KEY

本地測試：
pip install gradio plotly hopsworks yfinance
python app.py
"""

import os
import sys
import json
import warnings
from datetime import datetime, timedelta
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 匯入共享模組
sys.path.insert(0, os.path.dirname(__file__))
from src.constants import (
    UI_HISTORY_DAYS,
    FEATURE_GROUP_MAX_VERSION_SEARCH,
    GRADIO_SERVER_PORT,
    GRADIO_SERVER_NAME,
)
from src.utils import add_trading_days, next_trading_day

warnings.filterwarnings("ignore")
load_dotenv()

# ── 設定 ──────────────────────────────────────────────────────────
TICKER = os.environ.get("TICKER", "AAPL").upper()
HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY", "")
PREDICTION_GROUP = f"{TICKER.lower()}_predictions"
FEATURE_GROUP = f"{TICKER.lower()}_stock_features"
HISTORY_DAYS = UI_HISTORY_DAYS
# ─────────────────────────────────────────────────────────────────


def _validate_hopsworks_config() -> None:
    if not HOPSWORKS_PROJECT:
        raise ValueError("缺少 HOPSWORKS_PROJECT")
    if not HOPSWORKS_API_KEY:
        raise ValueError("缺少 HOPSWORKS_API_KEY")


# ── 資料讀取 ──────────────────────────────────────────────────────
def _hopsworks_login():
    import hopsworks
    _validate_hopsworks_config()
    return hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY
    )


def _get_latest_feature_group(fs, name: str, min_version: int = 1, max_version: int = FEATURE_GROUP_MAX_VERSION_SEARCH):
    """回傳最新可讀取的 Feature Group。"""
    for version in range(max_version, min_version - 1, -1):
        try:
            fg = fs.get_feature_group(name, version=version)
            if fg is None:
                continue
            return fg, version
        except Exception:
            continue
    raise RuntimeError(f"No available feature group found for {name}.")


def fetch_prediction(fs=None) -> dict:
    """讀取最新一筆預測結果"""
    hopsworks_error = None
    # 優先嘗試 Hopsworks
    try:
        if fs is None:
            project = _hopsworks_login()
            fs = project.get_feature_store()
        fg, _ = _get_latest_feature_group(fs, PREDICTION_GROUP)
        df = fg.read()
        if df is None or df.empty:
            raise RuntimeError(f"{PREDICTION_GROUP} 沒有可用資料。")
        df = df.sort_values("prediction_date", ascending=False)
        latest = df.iloc[0].to_dict()
        return latest
    except Exception as e:
        hopsworks_error = str(e)

    # Fallback：讀本地 JSON（開發用）
    local_path = "latest_prediction.json"
    if os.path.exists(local_path):
        with open(local_path) as f:
            return json.load(f)

    # 最後手段：用 yfinance 即時算一個假預測（示範用）
    return _mock_prediction(hopsworks_error)


def fetch_history(fs=None) -> pd.DataFrame:
    """讀取近期歷史收盤價 + 技術指標（直接用 yfinance 確保包含最新交易日）"""
    import yfinance as yf
    t = yf.Ticker(TICKER)
    df = t.history(period=f"{HISTORY_DAYS}d")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df["date"] = df["date"].astype(str)
    close = df["close"]
    df["ma_20"] = close.rolling(20).mean()
    df["ma_50"] = close.rolling(50).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    return df


def _mock_prediction(reason: str = None) -> dict:
    """示範用假資料（Hopsworks & 本地都沒資料時）"""
    import yfinance as yf
    t = yf.Ticker(TICKER)
    price = t.fast_info["last_price"]
    mock_pred = price * (1 + np.random.uniform(-0.01, 0.02))
    note = "⚠ 示範用模擬資料，非真實模型預測"
    if reason:
        note = f"{note}｜原因：{reason}"
    return {
        "ticker": TICKER,
        "prediction_date": str(datetime.today().date()),
        "current_close": round(price, 4),
        "predicted_close": round(mock_pred, 4),
        "change_pct": round((mock_pred - price) / price * 100, 4),
        "direction": "⬆ 看漲" if mock_pred > price else "⬇ 看跌",
        "predicted_at": datetime.now().isoformat(),
        "_note": note,
    }


# ── 圖表繪製 ──────────────────────────────────────────────────────
COLORS = {
    "bg": "#0d1117",
    "surface": "#161b22",
    "border": "#30363d",
    "text": "#e6edf3",
    "subtext": "#8b949e",
    "green": "#3fb950",
    "red": "#f85149",
    "blue": "#58a6ff",
    "yellow": "#d29922",
    "purple": "#bc8cff",
    "orange": "#ffa657",
}


def build_price_chart(df: pd.DataFrame, prediction: dict) -> go.Figure:
    """主圖：K 線 + MA + 預測點"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
        subplot_titles=("", "RSI (14)", "成交量"),
    )

    dates = df["date"].tolist()

    # ── K 線圖 ──
    fig.add_trace(go.Candlestick(
        x=dates,
        open=df["open"], close=df["close"],
        high=df["high"], low=df["low"],
        increasing_line_color=COLORS["green"],
        decreasing_line_color=COLORS["red"],
        name="K 線",
        showlegend=False,
    ), row=1, col=1)

    # ── MA20 / MA50 ──
    if "ma_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=df["ma_20"],
            mode="lines", name="MA20",
            line=dict(color=COLORS["blue"], width=1.2),
        ), row=1, col=1)

    if "ma_50" in df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=df["ma_50"],
            mode="lines", name="MA50",
            line=dict(color=COLORS["orange"], width=1.2),
        ), row=1, col=1)

    # ── 預測點（依目標天期）──
    base_date = str(prediction.get("prediction_date") or dates[-1])
    horizon_days = int(prediction.get("target_horizon_days", 1) or 1)
    pred_date = prediction.get("predicted_for_date") or _add_trading_days_str(base_date, horizon_days)
    pred_price = prediction["predicted_close"]
    curr_price = prediction["current_close"]
    is_up = pred_price >= curr_price
    pred_color = COLORS["green"] if is_up else COLORS["red"]

    # 虛線連接基準日→目標日預測
    fig.add_trace(go.Scatter(
        x=[base_date, pred_date],
        y=[curr_price, pred_price],
        mode="lines+markers",
        name="預測",
        line=dict(color=pred_color, width=2, dash="dot"),
        marker=dict(size=[0, 12], color=pred_color,
                    symbol="diamond",
                    line=dict(width=2, color=COLORS["bg"])),
    ), row=1, col=1)

    # 預測標籤
    fig.add_annotation(
        x=pred_date, y=pred_price,
        text=f" ${pred_price:.2f}<br> ({prediction['change_pct']:+.2f}%)",
        showarrow=False,
        font=dict(color=pred_color, size=13, family="monospace"),
        xanchor="left",
        row=1, col=1,
    )

    # ── RSI ──
    if "rsi_14" in df.columns:
        rsi = df["rsi_14"]
        fig.add_trace(go.Scatter(
            x=dates, y=rsi,
            mode="lines", name="RSI",
            line=dict(color=COLORS["purple"], width=1.5),
            showlegend=False,
        ), row=2, col=1)
        # 超買/超賣線
        for level, color in [(70, COLORS["red"]), (30, COLORS["green"])]:
            fig.add_hline(y=level, line_dash="dash",
                          line_color=color, opacity=0.5, row=2, col=1)

    # ── 成交量 ──
    if "volume" in df.columns:
        vol_colors = [
            COLORS["green"] if c >= o else COLORS["red"]
            for c, o in zip(df["close"], df["open"])
        ]
        fig.add_trace(go.Bar(
            x=dates, y=df["volume"],
            marker_color=vol_colors,
            name="成交量",
            showlegend=False,
            opacity=0.7,
        ), row=3, col=1)

    # ── 佈局 ──
    fig.update_layout(
        height=600,
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family="monospace"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", y=1.02,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
    )
    for i in range(1, 4):
        fig.update_xaxes(
            gridcolor=COLORS["border"], showgrid=True,
            zeroline=False, row=i, col=1,
        )
        fig.update_yaxes(
            gridcolor=COLORS["border"], showgrid=True,
            zeroline=False, row=i, col=1,
        )

    return fig


def _add_trading_days_str(date_str: str, days: int) -> str:
    """從指定日期往後推算 N 個交易日（返回字串）。"""
    from datetime import date
    d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    result = add_trading_days(d, days)
    return str(result)


# ── Gradio UI ─────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;500&display=swap');

body, .gradio-container {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
.header-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 16px;
}
.ticker-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: #8b949e;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.price-main {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 48px;
    font-weight: 600;
    color: #e6edf3;
    line-height: 1;
}
.price-change-up { color: #3fb950; font-size: 20px; font-family: monospace; }
.price-change-down { color: #f85149; font-size: 20px; font-family: monospace; }
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 20px;
}
.stat-card {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
}
.stat-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
.stat-value { font-family: 'IBM Plex Mono', monospace; font-size: 18px; color: #e6edf3; font-weight: 600; }
.footer-note { font-size: 11px; color: #484f58; margin-top: 16px; text-align: right; }
"""


def render_summary_html(prediction: dict) -> str:
    curr = prediction.get("current_close", 0)
    pred = prediction.get("predicted_close", 0)
    pct = prediction.get("change_pct", 0)
    prediction_date = str(prediction.get("prediction_date", ""))
    horizon_days = int(prediction.get("target_horizon_days", 1) or 1)
    forecast_date = prediction.get("predicted_for_date") or (_add_trading_days_str(prediction_date, horizon_days) if prediction_date else "—")
    is_up = pct >= 0
    arrow = "▲" if is_up else "▼"
    cls = "price-change-up" if is_up else "price-change-down"
    note = prediction.get("_note", "")
    updated = prediction.get("predicted_at", "")[:16].replace("T", " ")

    return f"""
<div class="header-card">
    <div class="ticker-label">{TICKER} · {horizon_days}-Trading-Day Close Prediction</div>
    <div class="price-main">${pred:,.2f}</div>
    <div class="{cls}">{arrow} {abs(pct):.2f}%&nbsp;&nbsp;from ${curr:,.2f}</div>
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-label">今日收盤</div>
            <div class="stat-value">${curr:,.2f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">預測目標價</div>
            <div class="stat-value" style="color:{'#3fb950' if is_up else '#f85149'}">${pred:,.2f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">預測方向</div>
            <div class="stat-value" style="color:{'#3fb950' if is_up else '#f85149'}">{prediction.get('direction','—')}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">資料日期</div>
            <div class="stat-value">{prediction_date or '—'}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">預測交易日</div>
            <div class="stat-value">{forecast_date}</div>
        </div>
    </div>
    {'<div style="margin-top:12px;padding:8px 12px;background:#2d1e00;border:1px solid #7d4e00;border-radius:6px;font-size:12px;color:#ffa657;">'+note+'</div>' if note else ''}
    <div class="footer-note">最後更新：{updated} UTC</div>
</div>
"""


def refresh_dashboard():
    """點擊「重新整理」時觸發"""
    fs = None
    try:
        project = _hopsworks_login()
        fs = project.get_feature_store()
    except Exception:
        pass

    try:
        prediction = fetch_prediction(fs)
        history = fetch_history(fs)
        chart = build_price_chart(history, prediction)
        summary = render_summary_html(prediction)
        status = "✅ 資料更新成功"
    except Exception as e:
        prediction = _mock_prediction()
        history = fetch_history(fs)
        chart = build_price_chart(history, prediction)
        summary = render_summary_html(prediction)
        status = f"⚠ 使用示範資料（{str(e)[:60]}）"

    return summary, chart, status


# ── 建立 Gradio App ───────────────────────────────────────────────
with gr.Blocks(css=CSS, title=f"{TICKER} Stock Predictor") as demo:

    gr.HTML(f"""
<div style="padding:20px 0 8px; display:flex; align-items:center; gap:12px;">
    <div style="font-family:'IBM Plex Mono',monospace; font-size:22px; font-weight:600; color:#e6edf3;">
        📈 {TICKER} Stock Predictor
    </div>
    <div style="font-size:12px; color:#8b949e; font-family:monospace;">
        Serverless ML · Hopsworks + Modal + HF Spaces
    </div>
</div>
""")

    summary_html = gr.HTML()
    chart_plot = gr.Plot(label="")
    status_text = gr.Textbox(
        label="狀態", interactive=False,
        elem_id="status-box",
    )

    refresh_btn = gr.Button("🔄 重新整理", variant="primary")
    refresh_btn.click(
        fn=refresh_dashboard,
        outputs=[summary_html, chart_plot, status_text],
    )

    # 頁面載入時自動執行一次
    demo.load(
        fn=refresh_dashboard,
        outputs=[summary_html, chart_plot, status_text],
    )

    gr.HTML("""
<div style="margin-top:24px; padding:16px; background:#161b22; border:1px solid #30363d;
    border-radius:8px; font-size:12px; color:#8b949e; line-height:1.8;">
    <strong style="color:#e6edf3;">免責聲明</strong><br>
    本工具僅供學習與研究目的，預測結果不構成任何投資建議。
    股票市場存在高度不確定性，過去表現不代表未來結果。
</div>
""")


if __name__ == "__main__":
    demo.launch(server_name=GRADIO_SERVER_NAME, server_port=GRADIO_SERVER_PORT)
