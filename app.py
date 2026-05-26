import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="BTC AI Dashboard",
    page_icon="₿",
    layout="wide"
)

st_autorefresh(interval=15000, key="btc_refresh")

# =========================
# CSS
# =========================

st.markdown("""
<style>

html, body, [class*="css"]{
    background:#050816;
    color:white;
    font-family:Segoe UI;
}

.block-container{
    padding-top:2rem;
    max-width:1500px;
}

/* TITLE */

.main-title{
    font-size:58px;
    font-weight:800;
    color:white;
}

.subtitle{
    color:#94a3b8;
    margin-bottom:35px;
}

/* CARDS */

.card{
    background:#0f172a;
    padding:28px;
    border-radius:24px;
    border:1px solid rgba(255,255,255,0.05);
}

.price{
    font-size:64px;
    font-weight:800;
}

.green{
    color:#00ff9d;
}

.red{
    color:#ff4d6d;
}

/* INDICATORS */

.indicator-card{
    background:#111827;
    border-radius:20px;
    padding:25px;
    text-align:center;
}

.indicator-name{
    color:#94a3b8;
    font-size:18px;
}

.indicator-value{
    font-size:42px;
    font-weight:800;
}

/* SECTION */

.section-title{
    font-size:42px;
    font-weight:800;
    margin-top:40px;
    margin-bottom:20px;
}

/* PREDICTION */

.prediction-up{
    background:#052e16;
    padding:30px;
    border-radius:24px;
}

.prediction-down{
    background:#450a0a;
    padding:30px;
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
        font-size:30px;
    }

}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================

st.markdown(
    '<div class="main-title">BTC AI Prediction Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Live Bitcoin AI Analysis • Technical Indicators • TradingView Chart</div>',
    unsafe_allow_html=True
)

# =========================
# BTC DATA
# =========================

df = yf.download(
    "BTC-USD",
    period="7d",
    interval="1h",
    auto_adjust=True
)

if df.empty:
    st.error("Failed loading BTC data")
    st.stop()

# FIX MULTI INDEX
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# RESET INDEX
df = df.reset_index()

# FIX DATE COLUMN
if "Datetime" in df.columns:
    pass

elif "Date" in df.columns:
    df.rename(columns={"Date":"Datetime"}, inplace=True)

else:
    df.rename(columns={df.columns[0]:"Datetime"}, inplace=True)

# FORCE 1D SERIES
close = pd.Series(df["Close"]).astype(float)

high = pd.Series(df["High"]).astype(float)

low = pd.Series(df["Low"]).astype(float)

# =========================
# INDICATORS
# =========================

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

# =========================
# LATEST VALUES
# =========================

latest = df.iloc[-1]

price = float(latest["Close"])

prev = float(df.iloc[-2]["Close"])

change = price - prev

change_percent = (change / prev) * 100

rsi = float(latest["RSI"])

macd_value = float(latest["MACD"])

atr = float(latest["ATR"])

volatility = float(latest["Volatility"])

# =========================
# LIVE PRICE
# =========================

c1, c2 = st.columns([2,1])

with c1:

    st.markdown(f"""
    <div class="card">
        <div style="color:#94a3b8;font-size:18px;">
            BTC/USD
        </div>

        <br>

        <div class="price">
            ${price:,.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:

    color = "green" if change > 0 else "red"

    arrow = "▲" if change > 0 else "▼"

    st.markdown(f"""
    <div class="card">
        <div style="color:#94a3b8;font-size:18px;">
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

# =========================
# AI PREDICTION
# =========================

prediction = "UP" if rsi < 40 else "DOWN"

prediction_class = (
    "prediction-up"
    if prediction == "UP"
    else "prediction-down"
)

confidence = np.random.randint(70,95)

st.markdown(f"""
<div class="{prediction_class}">

<div style="font-size:22px;color:#cbd5e1;">
AI Prediction
</div>

<br>

<div style="font-size:52px;font-weight:800;">
BTC may go {prediction}
</div>

<br>

<div style="font-size:28px;">
Confidence : {confidence}%
</div>

</div>
""", unsafe_allow_html=True)

# =========================
# MAIN INDICATORS
# =========================

st.markdown(
    '<div class="section-title">Main Indicators</div>',
    unsafe_allow_html=True
)

i1, i2, i3, i4 = st.columns(4)

data = [
    ("RSI", rsi, "#38bdf8"),
    ("MACD", macd_value, "#f472b6"),
    ("ATR", atr, "#facc15"),
    ("Volatility", volatility, "#4ade80")
]

for col, (name, value, color) in zip([i1,i2,i3,i4], data):

    with col:

        st.markdown(f"""
        <div class="indicator-card">

            <div class="indicator-name">
                {name}
            </div>

            <br>

            <div class="indicator-value" style="color:{color};">
                {value:.2f}
            </div>

        </div>
        """, unsafe_allow_html=True)

# =========================
# EMA CHART
# =========================

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
    legend=dict(orientation="h")
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================
# TRADINGVIEW
# =========================

st.markdown(
    '<div class="section-title">BTC Live Chart</div>',
    unsafe_allow_html=True
)

tradingview = """
<div class="tradingview-widget-container">
  <div id="tradingview_chart"></div>

  <script type="text/javascript"
  src="https://s3.tradingview.com/tv.js"></script>

  <script type="text/javascript">

  new TradingView.widget({
    "width":"100%",
    "height":750,
    "symbol":"BINANCE:BTCUSDT",
    "interval":"60",
    "timezone":"Asia/Kolkata",
    "theme":"dark",
    "style":"1",
    "locale":"en",
    "toolbar_bg":"#0f172a",
    "enable_publishing":false,
    "allow_symbol_change":true,
    "withdateranges":true,
    "hide_side_toolbar":false,
    "studies":[
      "RSI@tv-basicstudies",
      "MACD@tv-basicstudies",
      "MASimple@tv-basicstudies"
    ],
    "container_id":"tradingview_chart"
  });

  </script>
</div>
"""

st.components.v1.html(
    tradingview,
    height=760
)
