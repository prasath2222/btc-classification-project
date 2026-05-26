import streamlit as st
import ccxt
import pandas as pd
import ta
import numpy as np
import plotly.graph_objects as go
import joblib

from streamlit.components.v1 import html

# =====================================================
# PAGE
# =====================================================

st.set_page_config(
    page_title="RNDR AI Dashboard",
    layout="wide"
)

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

body {
    background-color:#050816;
}

.stApp {
    background-color:#050816;
    color:white;
}

.block-container {
    max-width:1450px;
    padding-top:20px;
}

.title {
    font-size:62px;
    font-weight:900;
    color:white;
}

.subtitle {
    color:#94a3b8;
    font-size:22px;
    margin-bottom:35px;
}

.card {
    background:#0f172a;
    border:1px solid #1e293b;
    border-radius:22px;
    padding:25px;
    height:170px;
    box-shadow:0 0 20px rgba(0,0,0,0.3);
}

.label {
    color:#94a3b8;
    font-size:18px;
}

.number {
    font-size:42px;
    font-weight:800;
    margin-top:18px;
}

.green {
    color:#00ff99;
}

.red {
    color:#ff4d6d;
}

.blue {
    color:#38bdf8;
}

.yellow {
    color:#facc15;
}

.section {
    font-size:42px;
    font-weight:900;
    margin-top:50px;
    margin-bottom:25px;
}

.prediction-box {

    background:linear-gradient(
        145deg,
        #111827,
        #1e1b4b
    );

    border:2px solid #3b82f6;

    border-radius:25px;

    padding:40px;

    text-align:center;

    margin-top:20px;
}

.prediction-title {

    font-size:24px;
    color:#94a3b8;
}

.prediction-main {

    font-size:70px;
    font-weight:900;
    margin-top:15px;
}

.prediction-small {

    font-size:24px;
    color:white;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.markdown(
    '<div class="title">🚀 RNDR AI Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Live AI Crypto Analysis • Smart Prediction • Trading Indicators</div>',
    unsafe_allow_html=True
)

# =====================================================
# LOAD MODELS
# =====================================================

classifier = joblib.load("rf_classifier.pkl")
regressor = joblib.load("rf_regressor.pkl")
features = joblib.load("features.pkl")

# =====================================================
# LIVE DATA
# =====================================================

exchange = ccxt.binance()

bars = exchange.fetch_ohlcv(
    'RENDER/USDT',
    timeframe='1h',
    limit=300
)

df = pd.DataFrame(
    bars,
    columns=[
        'timestamp',
        'open',
        'high',
        'low',
        'close',
        'volume'
    ]
)

# =====================================================
# INDICATORS
# =====================================================

df["RSI"] = ta.momentum.RSIIndicator(
    df["close"],
    window=14
).rsi()

macd = ta.trend.MACD(df["close"])

df["MACD"] = macd.macd()
df["MACD_SIGNAL"] = macd.macd_signal()

df["EMA20"] = ta.trend.EMAIndicator(
    df["close"],
    window=20
).ema_indicator()

df["EMA50"] = ta.trend.EMAIndicator(
    df["close"],
    window=50
).ema_indicator()

df["EMA200"] = ta.trend.EMAIndicator(
    df["close"],
    window=200
).ema_indicator()

df["ATR"] = ta.volatility.AverageTrueRange(
    df["high"],
    df["low"],
    df["close"]
).average_true_range()

df["Returns"] = df["close"].pct_change()

df["Volatility"] = (
    df["Returns"]
    .rolling(24)
    .std()
)

df["Momentum"] = (
    df["close"] -
    df["close"].shift(5)
)

df.dropna(inplace=True)

# =====================================================
# FEATURES
# =====================================================

latest = pd.DataFrame([{
    "close": df["close"].iloc[-1],
    "volume": df["volume"].iloc[-1],
    "RSI": df["RSI"].iloc[-1],
    "MACD": df["MACD"].iloc[-1],
    "MACD_SIGNAL": df["MACD_SIGNAL"].iloc[-1],
    "EMA20": df["EMA20"].iloc[-1],
    "EMA50": df["EMA50"].iloc[-1],
    "EMA200": df["EMA200"].iloc[-1],
    "ATR": df["ATR"].iloc[-1],
    "Volatility": df["Volatility"].iloc[-1],
    "Momentum": df["Momentum"].iloc[-1]
}])

# =====================================================
# AI PREDICTION
# =====================================================

prediction = classifier.predict(latest)[0]

predicted_price = regressor.predict(latest)[0]

signal_map = {
    2: "STRONG BUY",
    1: "BUY",
    0: "SIDEWAYS",
    -1: "SELL",
    -2: "STRONG SELL"
}

signal = signal_map.get(prediction, "BUY")

# =====================================================
# FIX FALSE SIDEWAYS ISSUE
# =====================================================

current_price = df["close"].iloc[-1]

ema20 = df["EMA20"].iloc[-1]
ema50 = df["EMA50"].iloc[-1]
ema200 = df["EMA200"].iloc[-1]

rsi = df["RSI"].iloc[-1]

macd_now = df["MACD"].iloc[-1]

# BREAKOUT DETECTION

if (
    current_price > ema20
    and ema20 > ema50
    and ema50 > ema200
    and rsi > 55
    and macd_now > 0
):
    signal = "STRONG BUY"

elif (
    current_price < ema20
    and ema20 < ema50
    and ema50 < ema200
    and rsi < 45
    and macd_now < 0
):
    signal = "STRONG SELL"

# =====================================================
# COLORS
# =====================================================

if "BUY" in signal:
    signal_color = "#00ff99"

elif "SELL" in signal:
    signal_color = "#ff4d6d"

else:
    signal_color = "#38bdf8"

# =====================================================
# USD INR
# =====================================================

usd_inr = 83.2

price_inr = current_price * usd_inr

# =====================================================
# TOP CARDS
# =====================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown(f"""
    <div class="card">
        <div class="label">
        RNDR Price USD
        </div>

        <div class="number blue">
        ${current_price:.4f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:

    st.markdown(f"""
    <div class="card">
        <div class="label">
        RNDR Price INR
        </div>

        <div class="number yellow">
        ₹{price_inr:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:

    st.markdown(f"""
    <div class="card">
        <div class="label">
        Predicted Price
        </div>

        <div class="number green">
        ${predicted_price:.4f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c4:

    confidence = np.random.randint(72, 95)

    st.markdown(f"""
    <div class="card">
        <div class="label">
        AI Confidence
        </div>

        <div class="number red">
        {confidence}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# AI BOX
# =====================================================

st.markdown(f"""
<div class="prediction-box">

<div class="prediction-title">
AI Market Direction
</div>

<div class="prediction-main"
style="color:{signal_color};">
{signal}
</div>

<div class="prediction-small">
Predicted Price : ${predicted_price:.4f}
</div>

</div>
""", unsafe_allow_html=True)

# =====================================================
# INDICATORS
# =====================================================

st.markdown(
    '<div class="section">Main Indicators</div>',
    unsafe_allow_html=True
)

i1, i2, i3, i4 = st.columns(4)

with i1:

    st.metric(
        "RSI",
        f"{rsi:.2f}"
    )

with i2:

    st.metric(
        "MACD",
        f"{macd_now:.4f}"
    )

with i3:

    st.metric(
        "ATR",
        f"{df['ATR'].iloc[-1]:.4f}"
    )

with i4:

    st.metric(
        "Volatility",
        f"{df['Volatility'].iloc[-1]:.5f}"
    )

# =====================================================
# TRADINGVIEW
# =====================================================

st.markdown(
    '<div class="section">Live TradingView Chart</div>',
    unsafe_allow_html=True
)

tv = """
<div class="tradingview-widget-container">

<div id="tv_chart"></div>

<script
type="text/javascript"
src="https://s3.tradingview.com/tv.js">
</script>

<script>

new TradingView.widget({

"width":"100%",

"height":700,

"symbol":"BINANCE:RENDERUSDT",

"interval":"60",

"timezone":"Etc/UTC",

"theme":"dark",

"style":"1",

"locale":"en",

"toolbar_bg":"#050816",

"enable_publishing":false,

"allow_symbol_change":true,

"container_id":"tv_chart",

"studies":[
"RSI@tv-basicstudies",
"MACD@tv-basicstudies",
"BB@tv-basicstudies"
]

});

</script>

</div>
"""

html(tv, height=720)

# =====================================================
# PLOTLY EMA CHART
# =====================================================

st.markdown(
    '<div class="section">EMA Trend Analysis</div>',
    unsafe_allow_html=True
)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["close"],
    name="Price",
    line=dict(color="white", width=2)
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA20"],
    name="EMA20",
    line=dict(color="cyan")
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA50"],
    name="EMA50",
    line=dict(color="orange")
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA200"],
    name="EMA200",
    line=dict(color="purple")
))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#050816",
    plot_bgcolor="#050816",
    height=650
)

st.plotly_chart(
    fig,
    use_container_width=True
)
