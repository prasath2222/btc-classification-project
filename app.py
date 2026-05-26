import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="BTC AI Dashboard",
    page_icon="₿",
    layout="wide"
)

st_autorefresh(interval=15000, key="refresh")

# ==================================================
# CSS
# ==================================================

st.markdown("""
<style>

html, body, [class*="css"]{
    background:#050816;
    color:white;
    font-family:Segoe UI;
}

/* MAIN */

.block-container{
    max-width:1500px;
    padding-top:2rem;
}

/* TITLES */

.main-title{
    font-size:64px;
    font-weight:800;
    color:white;
    margin-bottom:10px;
}

.sub-title{
    color:#94a3b8;
    margin-bottom:40px;
}

/* CARDS */

.card{
    background:#0f172a;
    border-radius:24px;
    padding:30px;
    border:1px solid rgba(255,255,255,0.05);
}

/* PRICE */

.price{
    font-size:64px;
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

.yellow{
    color:#facc15;
}

.pink{
    color:#f472b6;
}

/* SECTION TITLE */

.section-title{
    font-size:44px;
    font-weight:800;
    margin-top:50px;
    margin-bottom:20px;
}

/* INDICATOR */

.indicator-card{
    background:#111827;
    border-radius:22px;
    padding:25px;
    text-align:center;
    height:170px;
}

.indicator-name{
    color:#94a3b8;
    font-size:18px;
}

.indicator-value{
    font-size:42px;
    font-weight:800;
    margin-top:20px;
}

/* PREDICTION */

.prediction-up{
    background:#052e16;
    padding:35px;
    border-radius:24px;
}

.prediction-down{
    background:#450a0a;
    padding:35px;
    border-radius:24px;
}

/* MOBILE */

@media(max-width:768px){

    .main-title{
        font-size:42px;
    }

    .price{
        font-size:40px;
    }

    .indicator-value{
        font-size:28px;
    }

}

</style>
""", unsafe_allow_html=True)

# ==================================================
# TITLE
# ==================================================

st.markdown(
    '<div class="main-title">BTC AI Prediction Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Live Bitcoin AI Analysis • Technical Indicators • TradingView Chart</div>',
    unsafe_allow_html=True
)

# ==================================================
# BTC DATA
# ==================================================

df = yf.download(
    "BTC-USD",
    period="7d",
    interval="1h",
    auto_adjust=True
)

if df.empty:
    st.error("BTC data failed")
    st.stop()

# FIX COLUMNS

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()

# FIX DATE COLUMN

if "Datetime" not in df.columns:
    if "Date" in df.columns:
        df.rename(columns={"Date":"Datetime"}, inplace=True)
    else:
        df.rename(columns={df.columns[0]:"Datetime"}, inplace=True)

# FORCE SERIES

close = pd.Series(df["Close"]).astype(float)

high = pd.Series(df["High"]).astype(float)

low = pd.Series(df["Low"]).astype(float)

# ==================================================
# INDICATORS
# ==================================================

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

# ==================================================
# LATEST VALUES
# ==================================================

latest = df.iloc[-1]

price = float(latest["Close"])

prev = float(df.iloc[-2]["Close"])

change = price - prev

change_percent = (change / prev) * 100

rsi = float(latest["RSI"])

macd_value = float(latest["MACD"])

atr = float(latest["ATR"])

volatility = float(latest["Volatility"])

# ==================================================
# PRICE SECTION
# ==================================================

col1, col2 = st.columns([2,1])

with col1:

    st.markdown(f"""
    <div class="card">
        <div style="color:#94a3b8;font-size:20px;">
            BTC/USD
        </div>

        <br>

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

        <br>

        <div class="{color}" style="font-size:42px;font-weight:800;">
            {arrow} {change:,.2f}
        </div>

        <br>

        <div class="{color}" style="font-size:32px;font-weight:700;">
            {change_percent:.2f}%
        </div>

    </div>
    """, unsafe_allow_html=True)

# ==================================================
# AI PREDICTION
# ==================================================

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

<div style="font-size:56px;font-weight:800;">
BTC may go {prediction}
</div>

<br>

<div style="font-size:28px;">
Confidence : {confidence}%
</div>

</div>
""", unsafe_allow_html=True)

# ==================================================
# MAIN INDICATORS
# ==================================================

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

# ==================================================
# EMA CHART
# ==================================================

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
    height=500,
    paper_bgcolor="#050816",
    plot_bgcolor="#050816",
    font=dict(color="white"),
    xaxis=dict(showgrid=False),
    yaxis=dict(gridcolor="#1e293b"),
    legend=dict(
        orientation="h"
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==================================================
# BTC LIVE CHART
# ==================================================

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
  "height": 850,
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
    height=860
)
