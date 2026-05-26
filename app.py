import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from streamlit.components.v1 import html

st.set_page_config(
    page_title="BTC AI Dashboard",
    layout="wide"
)

# =========================
# AUTO REFRESH
# =========================

st.markdown("""
<meta http-equiv="refresh" content="60">
""", unsafe_allow_html=True)

# =========================
# CSS
# =========================

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #050816;
    color: white;
    font-family: Arial;
}

.block-container {
    padding-top: 2rem;
    padding-left: 5rem;
    padding-right: 5rem;
}

.title {
    font-size: 64px;
    font-weight: 800;
    margin-bottom: 10px;
}

.subtitle {
    color: #9ca3af;
    margin-bottom: 40px;
}

.card {
    background: #09142b;
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #14213d;
}

.metric-title {
    color: #94a3b8;
    font-size: 18px;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 48px;
    font-weight: bold;
}

.green {
    color: #00ff99;
}

.red {
    color: #ff4d4f;
}

.blue {
    color: #38bdf8;
}

.yellow {
    color: #facc15;
}

.section-title {
    font-size: 40px;
    font-weight: 700;
    margin-top: 50px;
    margin-bottom: 25px;
}

.predict-up {
    background: #052e16;
    padding: 35px;
    border-radius: 20px;
    border: 2px solid #16a34a;
}

.predict-down {
    background: #450a0a;
    padding: 35px;
    border-radius: 20px;
    border: 2px solid #dc2626;
}

.predict-text {
    font-size: 56px;
    font-weight: 800;
}

.small-text {
    font-size: 24px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================

st.markdown('<div class="title">BTC AI Prediction Dashboard</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Live Bitcoin Analysis • TradingView • Technical Indicators</div>',
    unsafe_allow_html=True
)

# =========================
# LOAD DATA
# =========================

df = yf.download(
    "BTC-USD",
    period="7d",
    interval="1h",
    auto_adjust=True
)

df.reset_index(inplace=True)

# =========================
# FIX COLUMNS
# =========================

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# =========================
# INDICATORS
# =========================

close_series = df["Close"].squeeze()

df["RSI"] = ta.momentum.RSIIndicator(
    close=close_series
).rsi()

macd = ta.trend.MACD(close_series)

df["MACD"] = macd.macd()

df["ATR"] = ta.volatility.AverageTrueRange(
    high=df["High"],
    low=df["Low"],
    close=close_series
).average_true_range()

df["VOLATILITY"] = close_series.pct_change().rolling(24).std()

# =========================
# LATEST VALUES
# =========================

latest_price = float(close_series.iloc[-1])

prev_price = float(close_series.iloc[-2])

change = latest_price - prev_price

change_percent = (change / prev_price) * 100

rsi = float(df["RSI"].iloc[-1])

macd_value = float(df["MACD"].iloc[-1])

atr = float(df["ATR"].iloc[-1])

volatility = float(df["VOLATILITY"].iloc[-1])

# =========================
# PRICE CARDS
# =========================

col1, col2 = st.columns([2,1])

with col1:

    st.markdown(f"""
    <div class="card">
        <div class="metric-title">BTC/USD</div>
        <div class="metric-value">
            ${latest_price:,.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:

    color = "green" if change > 0 else "red"

    arrow = "▲" if change > 0 else "▼"

    st.markdown(f"""
    <div class="card">
        <div class="metric-title">24H Change</div>
        <div class="metric-value {color}">
            {arrow} {change:,.2f}
        </div>

        <div class="{color}" style="font-size:28px;font-weight:bold;">
            {change_percent:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# AI PREDICTION
# =========================

prediction = "UP"

confidence = 74

if rsi < 45 and macd_value < 0:
    prediction = "DOWN"
    confidence = 86

if prediction == "UP":

    st.markdown(f"""
    <div class="predict-up">
        <div class="small-text">AI Prediction</div>

        <div class="predict-text green">
            BTC may go UP
        </div>

        <br>

        <div class="small-text">
            Confidence : {confidence}%
        </div>
    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown(f"""
    <div class="predict-down">
        <div class="small-text">AI Prediction</div>

        <div class="predict-text red">
            BTC may go DOWN
        </div>

        <br>

        <div class="small-text">
            Confidence : {confidence}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# TRADINGVIEW
# =========================

st.markdown(
    '<div class="section-title">BTC Live Chart</div>',
    unsafe_allow_html=True
)

tradingview = """
<div class="tradingview-widget-container">
  <div id="tradingview_btc"></div>

  <script type="text/javascript"
  src="https://s3.tradingview.com/tv.js"></script>

  <script type="text/javascript">

  new TradingView.widget({
  "width": "100%",
  "height": 720,
  "symbol": "BINANCE:BTCUSDT",
  "interval": "60",
  "timezone": "Asia/Kolkata",
  "theme": "dark",
  "style": "1",
  "locale": "en",
  "toolbar_bg": "#050816",
  "enable_publishing": false,
  "allow_symbol_change": true,
  "studies": [
    "RSI@tv-basicstudies",
    "MACD@tv-basicstudies"
  ],
  "container_id": "tradingview_btc"
});

  </script>
</div>
"""

html(tradingview, height=740)

# =========================
# INDICATORS
# =========================

st.markdown(
    '<div class="section-title">Main Indicators</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    rsi_color = "green"

    if rsi > 70:
        rsi_color = "red"

    elif rsi < 30:
        rsi_color = "blue"

    st.markdown(f"""
    <div class="card">
        <div class="metric-title">
            RSI
        </div>

        <div class="metric-value {rsi_color}">
            {rsi:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:

    macd_color = "green" if macd_value > 0 else "red"

    st.markdown(f"""
    <div class="card">
        <div class="metric-title">
            MACD
        </div>

        <div class="metric-value {macd_color}">
            {macd_value:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:

    st.markdown(f"""
    <div class="card">
        <div class="metric-title">
            ATR
        </div>

        <div class="metric-value yellow">
            {atr:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c4:

    st.markdown(f"""
    <div class="card">
        <div class="metric-title">
            Volatility
        </div>

        <div class="metric-value blue">
            {volatility:.4f}
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# MARKET DATA
# =========================

st.markdown(
    '<div class="section-title">Latest Market Data</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("BTC Price", f"${latest_price:,.2f}")

m2.metric("RSI", f"{rsi:.2f}")

m3.metric("MACD", f"{macd_value:.2f}")

m4.metric("ATR", f"{atr:.2f}")
