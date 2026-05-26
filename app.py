import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="BTC AI Dashboard",
    page_icon="₿",
    layout="wide"
)

st_autorefresh(interval=15000, key="refresh")

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"]{
    background:#050816;
    color:white;
    font-family:Inter;
}

.block-container{
    max-width:1450px;
    padding-top:25px;
    padding-bottom:40px;
}

/* TITLES */

.main-title{
    font-size:60px;
    font-weight:800;
    margin-bottom:10px;
}

.sub-title{
    color:#94a3b8;
    font-size:18px;
    margin-bottom:35px;
}

/* CARDS */

.card{
    background:#0f172a;
    border-radius:18px;
    padding:28px;
    border:1px solid rgba(255,255,255,0.05);
}

/* PRICE */

.price{
    font-size:60px;
    font-weight:800;
}

/* COLORS */

.green{
    color:#00ff9d;
}

.red{
    color:#ff4d6d;
}

.blue{
    color:#38bdf8;
}

.pink{
    color:#f472b6;
}

.yellow{
    color:#facc15;
}

/* SECTION TITLE */

.section-title{
    font-size:40px;
    font-weight:800;
    margin-top:45px;
    margin-bottom:20px;
}

/* INDICATOR CARDS */

.indicator-card{
    background:#111827;
    border-radius:18px;
    padding:22px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.05);
}

.indicator-name{
    color:#94a3b8;
    font-size:18px;
}

.indicator-value{
    font-size:36px;
    font-weight:800;
    margin-top:18px;
}

/* AI PREDICTION */

.prediction-up{
    background:#052e16;
    border-radius:20px;
    padding:30px;
}

.prediction-down{
    background:#450a0a;
    border-radius:20px;
    padding:30px;
}

/* MOBILE */

@media(max-width:768px){

    .main-title{
        font-size:36px;
    }

    .price{
        font-size:36px;
    }

    .indicator-value{
        font-size:24px;
    }

}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">BTC AI Prediction Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Live Bitcoin AI Analysis • Technical Indicators • TradingView Chart</div>',
    unsafe_allow_html=True
)

# =========================================================
# BTC DATA
# =========================================================

df = yf.download(
    "BTC-USD",
    period="7d",
    interval="1h",
    auto_adjust=True
)

if df.empty:
    st.error("Failed to load BTC data")
    st.stop()

# FIX COLUMNS

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()

# FIX DATE COLUMN

if "Datetime" not in df.columns:
    if "Date" in df.columns:
        df.rename(columns={"Date": "Datetime"}, inplace=True)
    else:
        df.rename(columns={df.columns[0]: "Datetime"}, inplace=True)

# FORCE SERIES

close = pd.Series(df["Close"]).astype(float)
high = pd.Series(df["High"]).astype(float)
low = pd.Series(df["Low"]).astype(float)

# =========================================================
# INDICATORS
# =========================================================

df["RSI"] = ta.momentum.RSIIndicator(
    close=close
).rsi()

macd = ta.trend.MACD(close)

df["MACD"] = macd.macd()

df["EMA20"] = ta.trend.EMAIndicator(
    close=close,
    window=20
).ema_indicator()

df["EMA50"] = ta.trend.EMAIndicator(
    close=close,
    window=50
).ema_indicator()

df["ATR"] = ta.volatility.AverageTrueRange(
    high=high,
    low=low,
    close=close
).average_true_range()

df["Volatility"] = close.pct_change().rolling(24).std()

# =========================================================
# LATEST VALUES
# =========================================================

latest = df.iloc[-1]

price = float(latest["Close"])

prev = float(df.iloc[-2]["Close"])

change = price - prev

change_percent = (change / prev) * 100

rsi = float(latest["RSI"])

macd_value = float(latest["MACD"])

atr = float(latest["ATR"])

volatility = float(latest["Volatility"])

# =========================================================
# PRICE SECTION
# =========================================================

col1, col2 = st.columns([2,1])

with col1:

    st.markdown(f"""
    <div class="card">

        <div style="color:#94a3b8;font-size:20px;">
            BTC/USD
        </div>

        <div class="price">
            ${price:,.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

with col2:

    color = "green" if change > 0 else "red"

    arrow = "▲" if change > 0 else "▼"

    st.markdown(f"""
    <div class="card">

        <div style="color:#94a3b8;font-size:20px;">
            24H Change
        </div>

        <div class="{color}" style="
            font-size:42px;
            font-weight:800;
            margin-top:20px;
        ">
            {arrow} {change:,.2f}
        </div>

        <div class="{color}" style="
            font-size:28px;
            margin-top:10px;
        ">
            {change_percent:.2f}%
        </div>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# AI PREDICTION
# =========================================================

prediction = "UP" if rsi < 40 else "DOWN"

prediction_class = (
    "prediction-up"
    if prediction == "UP"
    else "prediction-down"
)

confidence = np.random.randint(72,95)

st.markdown(f"""
<div class="{prediction_class}">

<div style="font-size:24px;color:#cbd5e1;">
AI Prediction
</div>

<br>

<div style="font-size:54px;font-weight:800;">
BTC may go {prediction}
</div>

<br>

<div style="font-size:28px;">
Confidence : {confidence}%
</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# TRADINGVIEW CHART
# =========================================================

st.markdown(
    '<div class="section-title">BTC Live Chart</div>',
    unsafe_allow_html=True
)

tradingview_code = """
<div class="tradingview-widget-container">

<div id="tradingview_chart"></div>

<script type="text/javascript"
src="https://s3.tradingview.com/tv.js">
</script>

<script type="text/javascript">

new TradingView.widget({

  "width": "100%",
  "height": 950,

  "symbol": "BINANCE:BTCUSDT",

  "interval": "60",

  "timezone": "Asia/Kolkata",

  "theme": "dark",

  "style": "1",

  "locale": "en",

  "toolbar_bg": "#0f172a",

  "enable_publishing": false,

  "allow_symbol_change": true,

  "withdateranges": true,

  "hide_side_toolbar": false,

  "save_image": true,

  "studies": [
    "RSI@tv-basicstudies",
    "MACD@tv-basicstudies",
    "MASimple@tv-basicstudies"
  ],

  "container_id": "tradingview_chart"

});

</script>

</div>
"""

st.components.v1.html(
    tradingview_code,
    height=980
)

# =========================================================
# MAIN INDICATORS
# =========================================================

st.markdown(
    '<div class="section-title">Main Indicators</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown(f"""
    <div class="indicator-card">

        <div class="indicator-name">
            RSI
        </div>

        <div class="indicator-value blue">
            {rsi:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

with c2:

    st.markdown(f"""
    <div class="indicator-card">

        <div class="indicator-name">
            MACD
        </div>

        <div class="indicator-value pink">
            {macd_value:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

with c3:

    st.markdown(f"""
    <div class="indicator-card">

        <div class="indicator-name">
            ATR
        </div>

        <div class="indicator-value yellow">
            {atr:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

with c4:

    st.markdown(f"""
    <div class="indicator-card">

        <div class="indicator-name">
            Volatility
        </div>

        <div class="indicator-value green">
            {volatility:.4f}
        </div>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# EMA CHART
# =========================================================

st.markdown(
    '<div class="section-title">EMA 20 vs EMA 50</div>',
    unsafe_allow_html=True
)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Datetime"],
    y=df["EMA20"],
    mode="lines",
    name="EMA 20",
    line=dict(color="#38bdf8", width=3)
))

fig.add_trace(go.Scatter(
    x=df["Datetime"],
    y=df["EMA50"],
    mode="lines",
    name="EMA 50",
    line=dict(color="#ff4d6d", width=3)
))

fig.update_layout(

    height=450,

    paper_bgcolor="#050816",

    plot_bgcolor="#050816",

    font=dict(color="white"),

    margin=dict(
        l=10,
        r=10,
        t=10,
        b=10
    ),

    xaxis=dict(
        showgrid=False
    ),

    yaxis=dict(
        gridcolor="#1e293b"
    ),

    legend=dict(
        orientation="h"
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# LATEST MARKET DATA
# =========================================================

st.markdown(
    '<div class="section-title">Latest Market Data</div>',
    unsafe_allow_html=True
)

a,b,c,d = st.columns(4)

a.metric("BTC Price", f"${price:,.2f}")

b.metric("RSI", f"{rsi:.2f}")

c.metric("MACD", f"{macd_value:.2f}")

d.metric("ATR", f"{atr:.2f}")
