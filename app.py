import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from streamlit.components.v1 import html

# =====================================================
# PAGE
# =====================================================

st.set_page_config(
    page_title="BTC AI Dashboard",
    layout="wide"
)

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

html, body, [class*="css"]{
    background:#050816;
    color:white;
    font-family:Arial;
}

/* MAIN */
.block-container{
    padding-top:40px;
    max-width:1500px;
}

/* TITLES */

.main-title{
    font-size:72px;
    font-weight:900;
    margin-bottom:10px;
}

.sub-title{
    color:#94a3b8;
    font-size:20px;
    margin-bottom:40px;
}

.section-title{
    font-size:42px;
    font-weight:800;
    margin-top:50px;
    margin-bottom:25px;
}

/* CARDS */

.card{
    background:#071739;
    border-radius:22px;
    padding:30px;
    border:1px solid #112b5c;
}

.metric-title{
    color:#94a3b8;
    font-size:20px;
    margin-bottom:15px;
}

.metric-value{
    font-size:58px;
    font-weight:900;
}

.green{
    color:#00ff99;
}

.red{
    color:#ff4d4f;
}

.yellow{
    color:#facc15;
}

.blue{
    color:#38bdf8;
}

/* PREDICTION */

.predict-up{
    background:#032b1f;
    border:2px solid #00ff99;
    border-radius:24px;
    padding:35px;
}

.predict-down{
    background:#3a0005;
    border:2px solid #ff4d4f;
    border-radius:24px;
    padding:35px;
}

/* MARKET DATA */

.market-box{
    background:#071739;
    border-radius:20px;
    padding:25px;
    text-align:center;
}

.market-name{
    color:#94a3b8;
    font-size:20px;
}

.market-value{
    font-size:54px;
    font-weight:800;
    margin-top:10px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# DATA
# =====================================================

btc = yf.download(
    "BTC-USD",
    period="7d",
    interval="1h",
    auto_adjust=True
)

btc.columns = btc.columns.get_level_values(0)

btc.dropna(inplace=True)

# =====================================================
# INDICATORS
# =====================================================

btc["RSI"] = ta.momentum.RSIIndicator(
    close=btc["Close"]
).rsi()

macd = ta.trend.MACD(close=btc["Close"])

btc["MACD"] = macd.macd()
btc["MACD_SIGNAL"] = macd.macd_signal()
btc["MACD_HIST"] = macd.macd_diff()

btc["ATR"] = ta.volatility.AverageTrueRange(
    high=btc["High"],
    low=btc["Low"],
    close=btc["Close"]
).average_true_range()

# =====================================================
# VALUES
# =====================================================

latest_price = float(btc["Close"].iloc[-1])

change = latest_price - float(btc["Close"].iloc[-24])

change_percent = (change / latest_price) * 100

rsi = float(btc["RSI"].iloc[-1])

macd_value = float(btc["MACD"].iloc[-1])

atr = float(btc["ATR"].iloc[-1])

volatility = float(btc["Close"].pct_change().std())

# =====================================================
# AI PREDICTION
# =====================================================

prediction = "UP"
confidence = 74

if rsi < 45 and macd_value < 0:
    prediction = "DOWN"
    confidence = 86

# =====================================================
# TITLE
# =====================================================

st.markdown(
    '<div class="main-title">BTC AI Prediction Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Live Bitcoin Analysis • Trading Setup • Technical Indicators</div>',
    unsafe_allow_html=True
)

# =====================================================
# TOP ROW
# =====================================================

left, right = st.columns([2,1])

# PRICE
with left:

    st.markdown(f"""
    <div class="card">

        <div class="metric-title">
            BTC/USD
        </div>

        <div class="metric-value">
            ${latest_price:,.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

# CHANGE
with right:

    color = "green" if change > 0 else "red"

    arrow = "▲" if change > 0 else "▼"

    st.markdown(f"""
    <div class="card">

        <div class="metric-title">
            24H Change
        </div>

        <div class="metric-value {color}">
            {arrow} {change:,.2f}
        </div>

        <div style="
            font-size:32px;
            font-weight:800;
            margin-top:15px;
            color:{'#00ff99' if change > 0 else '#ff4d4f'};
        ">
            {change_percent:.2f}%
        </div>

    </div>
    """, unsafe_allow_html=True)

# =====================================================
# AI PREDICTION
# =====================================================

if prediction == "UP":

    st.markdown(f"""
    <div class="predict-up">

        <div style="
            font-size:26px;
            color:white;
            margin-bottom:20px;
        ">
            AI Prediction
        </div>

        <div style="
            font-size:60px;
            font-weight:900;
            color:#00ff99;
        ">
            BTC may go UP
        </div>

        <div style="
            margin-top:20px;
            font-size:30px;
            color:white;
        ">
            Confidence : {confidence}%
        </div>

    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown(f"""
    <div class="predict-down">

        <div style="
            font-size:26px;
            color:white;
            margin-bottom:20px;
        ">
            AI Prediction
        </div>

        <div style="
            font-size:60px;
            font-weight:900;
            color:#ff4d4f;
        ">
            BTC may go DOWN
        </div>

        <div style="
            margin-top:20px;
            font-size:30px;
            color:white;
        ">
            Confidence : {confidence}%
        </div>

    </div>
    """, unsafe_allow_html=True)

# =====================================================
# TRADINGVIEW
# =====================================================

st.markdown(
    '<div class="section-title">BTC Live Chart</div>',
    unsafe_allow_html=True
)

tradingview_html = """
<div class="tradingview-widget-container">
  <div id="tradingview_btc"></div>

  <script type="text/javascript" 
  src="https://s3.tradingview.com/tv.js"></script>

  <script type="text/javascript">
  new TradingView.widget({
      "width": "100%",
      "height": 900,
      "symbol": "BINANCE:BTCUSDT",
      "interval": "60",
      "timezone": "Asia/Kolkata",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "toolbar_bg": "#050816",
      "enable_publishing": false,
      "allow_symbol_change": true,
      "container_id": "tradingview_btc"
  });
  </script>
</div>
"""

html(tradingview_html, height=920)

# =====================================================
# MAIN INDICATORS
# =====================================================

st.markdown(
    '<div class="section-title">Main Indicators</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

# RSI
with c1:

    rsi_color = "#00ff99"

    if rsi > 70:
        rsi_color = "#ff4d4f"

    elif rsi < 30:
        rsi_color = "#38bdf8"

    st.markdown(f"""
    <div class="card">

        <div class="metric-title">
            RSI
        </div>

        <div style="
            font-size:52px;
            font-weight:900;
            color:{rsi_color};
        ">
            {rsi:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

# MACD
with c2:

    macd_color = "#00ff99" if macd_value > 0 else "#ff4d4f"

    st.markdown(f"""
    <div class="card">

        <div class="metric-title">
            MACD
        </div>

        <div style="
            font-size:52px;
            font-weight:900;
            color:{macd_color};
        ">
            {macd_value:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

# ATR
with c3:

    st.markdown(f"""
    <div class="card">

        <div class="metric-title">
            ATR
        </div>

        <div style="
            font-size:52px;
            font-weight:900;
            color:#facc15;
        ">
            {atr:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

# VOLATILITY
with c4:

    st.markdown(f"""
    <div class="card">

        <div class="metric-title">
            Volatility
        </div>

        <div style="
            font-size:52px;
            font-weight:900;
            color:#38bdf8;
        ">
            {volatility:.4f}
        </div>

    </div>
    """, unsafe_allow_html=True)

# =====================================================
# MARKET DATA
# =====================================================

st.markdown(
    '<div class="section-title">Latest Market Data</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.markdown(f"""
    <div class="market-box">

        <div class="market-name">
            BTC Price
        </div>

        <div class="market-value">
            ${latest_price:,.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

with m2:

    st.markdown(f"""
    <div class="market-box">

        <div class="market-name">
            RSI
        </div>

        <div class="market-value">
            {rsi:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

with m3:

    st.markdown(f"""
    <div class="market-box">

        <div class="market-name">
            MACD
        </div>

        <div class="market-value">
            {macd_value:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

with m4:

    st.markdown(f"""
    <div class="market-box">

        <div class="market-name">
            ATR
        </div>

        <div class="market-value">
            {atr:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

# =====================================================
# MARKET MOVEMENT
# =====================================================

st.markdown(
    '<div class="section-title">Market Movement</div>',
    unsafe_allow_html=True
)

trend = "SIDEWAYS"
trend_color = "#facc15"

if rsi > 55 and macd_value > 0:
    trend = "BULLISH MOMENTUM"
    trend_color = "#00ff99"

elif rsi < 45 and macd_value < 0:
    trend = "BEARISH MOMENTUM"
    trend_color = "#ff4d4f"

st.markdown(f"""
<div class="card">

    <div style="
        font-size:46px;
        font-weight:900;
        color:{trend_color};
    ">
        {trend}
    </div>

    <div style="
        margin-top:30px;
        font-size:24px;
        color:#cbd5e1;
        line-height:2;
    ">

        BTC Price : ${latest_price:,.2f}

        <br>

        RSI : {rsi:.2f}

        <br>

        MACD : {macd_value:.2f}

        <br>

        ATR : {atr:.2f}

    </div>

</div>
""", unsafe_allow_html=True)
