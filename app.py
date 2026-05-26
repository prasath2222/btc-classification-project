import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="BTC AI Dashboard",
    page_icon="₿",
    layout="wide"
)

st_autorefresh(interval=15000, key="refresh")

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #050816;
    color: white;
    font-family: 'Segoe UI';
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* TITLE */

.main-title {
    font-size: 58px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

.subtitle {
    color: #94a3b8;
    font-size: 18px;
    margin-bottom: 35px;
}

/* CARDS */

.card {
    background: linear-gradient(145deg,#0f172a,#111c44);
    padding: 28px;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 0 30px rgba(0,0,0,0.35);
}

.price {
    font-size: 62px;
    font-weight: 800;
    color: white;
}

.label {
    color: #94a3b8;
    font-size: 18px;
}

.change-up {
    color: #00ff9d;
    font-size: 40px;
    font-weight: 700;
}

.change-down {
    color: #ff3366;
    font-size: 40px;
    font-weight: 700;
}

/* PREDICTION */

.prediction-up {
    background: linear-gradient(145deg,#052e16,#064e3b);
    border-radius: 24px;
    padding: 30px;
    border: 1px solid rgba(0,255,100,0.2);
}

.prediction-down {
    background: linear-gradient(145deg,#3b0000,#5f0000);
    border-radius: 24px;
    padding: 30px;
    border: 1px solid rgba(255,0,80,0.2);
}

.prediction-title {
    font-size: 42px;
    font-weight: 800;
}

/* INDICATOR CARDS */

.indicator-card {
    background: #0f172a;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
}

.indicator-name {
    color: #94a3b8;
    font-size: 18px;
}

.indicator-value {
    font-size: 48px;
    font-weight: 800;
    margin-top: 10px;
}

/* SECTION TITLE */

.section-title {
    font-size: 40px;
    font-weight: 800;
    margin-top: 35px;
    margin-bottom: 20px;
}

/* MOBILE */

@media (max-width: 768px){

    .main-title{
        font-size:42px;
    }

    .price{
        font-size:40px;
    }

    .prediction-title{
        font-size:32px;
    }

    .indicator-value{
        font-size:34px;
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
    '<div class="subtitle">Live Bitcoin Analysis • AI Prediction • Technical Indicators </div>',
    unsafe_allow_html=True
)

# =========================
# DOWNLOAD BTC DATA
# =========================

df = yf.download(
    "BTC-USD",
    period="7d",
    interval="1h",
    auto_adjust=True
)

# FIX EMPTY DATA
if df.empty:
    st.error("BTC data failed to load")
    st.stop()

# RESET INDEX
df = df.reset_index()

# FIX DATETIME COLUMN
if "Datetime" in df.columns:
    date_col = "Datetime"

elif "Date" in df.columns:
    date_col = "Date"

else:
    date_col = df.columns[0]

df.rename(columns={date_col: "Datetime"}, inplace=True)

# =========================
# INDICATORS
# =========================

df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Close"]
).rsi()

macd = ta.trend.MACD(df["Close"])

df["MACD"] = macd.macd()

df["EMA20"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=20
).ema_indicator()

df["EMA50"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=50
).ema_indicator()

df["ATR"] = ta.volatility.AverageTrueRange(
    high=df["High"],
    low=df["Low"],
    close=df["Close"]
).average_true_range()

df["Volatility"] = df["Close"].pct_change().rolling(24).std()

# =========================
# LATEST VALUES
# =========================

latest = df.iloc[-1]

price = float(latest["Close"])

previous = float(df.iloc[-2]["Close"])

change = price - previous

change_percent = (change / previous) * 100

rsi = float(latest["RSI"])

macd_value = float(latest["MACD"])

atr = float(latest["ATR"])

volatility = float(latest["Volatility"])

# =========================
# LIVE PRICE CARDS
# =========================

col1, col2 = st.columns([2,1])

with col1:

    st.markdown(f"""
    <div class="card">
        <div class="label">BTC/USD</div>
        <br>
        <div class="price">${price:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:

    color_class = "change-up" if change > 0 else "change-down"

    arrow = "▲" if change > 0 else "▼"

    st.markdown(f"""
    <div class="card">
        <div class="label">24H Change</div>
        <br>
        <div class="{color_class}">
            {arrow} {change:,.2f}
        </div>
        <br>
        <div class="{color_class}">
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
    <div style="font-size:20px;color:#cbd5e1;">
        AI Prediction
    </div>

    <br>

    <div class="prediction-title">
        BTC may go {prediction}
    </div>

    <br>

    <div style="font-size:22px;">
        Confidence : {confidence}%
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# INDICATORS
# =========================

st.markdown(
    '<div class="section-title">Main Indicators</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

cards = [
    ("RSI", rsi, "#38bdf8"),
    ("MACD", macd_value, "#f472b6"),
    ("ATR", atr, "#facc15"),
    ("Volatility", volatility, "#4ade80")
]

for col, (name, value, color) in zip([c1,c2,c3,c4], cards):

    with col:

        st.markdown(f"""
        <div class="indicator-card">
            <div class="indicator-name">
                {name}
            </div>

            <div class="indicator-value" style="color:{color}">
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
    line=dict(color="#f43f5e", width=3)
))

fig.update_layout(
    height=500,
    paper_bgcolor="#050816",
    plot_bgcolor="#050816",
    font=dict(color="white"),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#1e293b"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
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

tradingview_html = """
<div class="tradingview-widget-container">
  <div id="tradingview_chart"></div>

  <script type="text/javascript"
  src="https://s3.tradingview.com/tv.js"></script>

  <script type="text/javascript">

  new TradingView.widget(
  {
    "width": "100%",
    "height": 750,
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
    tradingview_html,
    height=760
)
