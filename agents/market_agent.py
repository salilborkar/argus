"""
agents/market_agent.py
-----------------------
Market Data & Chart Agent for Argus.

Responsibility:
  - Fetch historical price data from Yahoo Finance
  - Generate a chart with closing price, 20-day MA, 50-day MA, and volume
  - Date range is SET BY THE ORCHESTRATOR based on sentiment output — not hardcoded
  - Classify the price trend over that period
  - Write results into ResearchSession

Input (reads from session):
  - session.ticker
  - session.date_range_start  (set by orchestrator)
  - session.date_range_end    (set by orchestrator)

Output (writes to session):
  - session.chart_path
  - session.price_trend
  - session.price_change_pct
"""

import os
import yfinance as yf
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — works in both CLI and Streamlit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime

from core.session import ResearchSession
from config import CHARTS_DIR, CHART_FIGSIZE


def _classify_trend(df: pd.DataFrame) -> str:
    """Classify price trend based on first/last close and 20-day MA slope."""
    if df.empty or len(df) < 5:
        return "insufficient data"
    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]
    change_pct = ((end_price - start_price) / start_price) * 100

    if change_pct > 5:
        return "uptrend"
    elif change_pct < -5:
        return "downtrend"
    else:
        return "sideways"


def _generate_chart(ticker: str, df: pd.DataFrame, session_id: str) -> str:
    """
    Generate a 2-panel chart: price + MAs on top, volume on bottom.
    Saves to charts/ and returns the file path.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)

    # Calculate moving averages
    df = df.copy()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=CHART_FIGSIZE,
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor("#0f1117")

    for ax in (ax1, ax2):
        ax.set_facecolor("#0f1117")
        ax.tick_params(colors="#cccccc", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    # --- Price panel ---
    ax1.plot(df.index, df["Close"], color="#00bfff", linewidth=1.8, label="Close Price", zorder=3)
    ax1.plot(df.index, df["MA20"], color="#ff9800", linewidth=1.2, linestyle="--", label="20-day MA", zorder=2)
    ax1.plot(df.index, df["MA50"], color="#ab47bc", linewidth=1.2, linestyle="--", label="50-day MA", zorder=2)
    ax1.fill_between(df.index, df["Close"], df["Close"].min(), alpha=0.08, color="#00bfff")

    start_str = df.index[0].strftime("%b %d, %Y")
    end_str = df.index[-1].strftime("%b %d, %Y")
    ax1.set_title(
        f"{ticker}  |  {start_str} → {end_str}",
        color="#ffffff", fontsize=13, fontweight="bold", pad=12,
    )
    ax1.set_ylabel("Price (USD)", color="#cccccc", fontsize=10)
    ax1.legend(
        facecolor="#1a1a2e", edgecolor="#333333",
        labelcolor="#cccccc", fontsize=9, loc="upper left",
    )
    ax1.grid(True, linestyle="--", alpha=0.2, color="#555555")
    ax1.yaxis.label.set_color("#cccccc")

    # --- Volume panel ---
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["Close"], df["Open"])]
    ax2.bar(df.index, df["Volume"], color=colors, alpha=0.7, width=1.5)
    ax2.set_ylabel("Volume", color="#cccccc", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.15, color="#555555")
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
    )

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", color="#cccccc")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    chart_path = os.path.join(CHARTS_DIR, f"{session_id}_{ticker}.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    return chart_path


def run(session: ResearchSession):
    """
    Main entry point called by the orchestrator.
    Reads session.ticker + date range (set by orchestrator), writes chart/trend fields.
    """
    ticker = session.ticker
    print(f"  [market_agent] Fetching price data for {ticker}...")

    # Parse date range set by orchestrator
    start = session.date_range_start
    end = session.date_range_end or datetime.today().strftime("%Y-%m-%d")

    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as e:
        session.escalation_flags.append(f"market_agent: yfinance download failed — {e}")
        return

    if df is None or df.empty:
        session.escalation_flags.append(f"market_agent: no price data returned for {ticker}")
        return

    # Flatten MultiIndex columns if present (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure required columns exist
    required = {"Open", "Close", "Volume"}
    if not required.issubset(df.columns):
        session.escalation_flags.append(
            f"market_agent: missing columns in yfinance data. Got: {list(df.columns)}"
        )
        return

    # Classify trend
    session.price_trend = _classify_trend(df)
    start_price = float(df["Close"].iloc[0])
    end_price = float(df["Close"].iloc[-1])
    session.price_change_pct = round(((end_price - start_price) / start_price) * 100, 2)

    # Generate chart
    chart_path = _generate_chart(ticker, df, session.session_id)
    session.chart_path = chart_path

    print(
        f"  [market_agent] Done. Trend: {session.price_trend}, "
        f"Change: {session.price_change_pct:+.1f}%, Chart: {chart_path}"
    )
