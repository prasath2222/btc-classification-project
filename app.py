import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import pickle
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="BTC AI Dashboard",
    layout="wide"
)

# =========================================================
# AUTO REFRESH
# =========================================================

st_autorefresh(
    interval=60000,
    key="btc_refresh"
)

# =========================================================
# DARK THEME CSS
# =========================================================

st.markdown("""
<style>

body {
    background-color: #050816;
    color: white;
}

.stApp {
    background-color: #050816;
}

.main-title {
    font-size: 52px;
    font-weight: bold;
    color: white;
}

.card {
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 18px;
}

.green-card {
    background-color: rgba(0,255,100,0.15);
    border: 1px solid rgba(0,255,100,0.4);
}

.red-card {
    background-color: rgba(255,0,80,0.15);
    border: 1px solid rgba(255,0,80,0.4);
}

.blue-card {
    background-color: rgba(0,120,255,0.15);
    border: 1px solid rgba(0,120,255,0.4);
}

.indicator-title {
    font-size: 22px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<p class="main-title">BTC AI Prediction Dashboard</p>',
    unsafe_allow_html=True
)

# =========================================================
# LOAD MODEL
# =========================================================

model = pickle.load(
    open("btc_model.pkl", "rb")
)

# =========================================================
# DOWNLOAD BTC DATA
# =========================================================

df = yf.download(
    "BTC-USD",
    period="60d",
    interval="1h",
    auto_adjust=True
)

# =========================================================
# FIX MULTI INDEX
# =========================================================

if isinstance(df.columns, pd.MultiIndex):

    df.columns = df.columns.get_level_values(0)

# =========================================================
# RESET INDEX
# =========================================================

df = df.reset_index()

# =========================================================
# RSI
# =========================================================

df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Close"],
    window=14
).rsi()

# =========================================================
# MACD
# =========================================================

macd = ta.trend.MACD(
    close=df["Close"]
)

df["MACD"] = macd.macd()

df["MACD_SIGNAL"] = macd.macd_signal()

df["MACD_DIFF"] = macd.macd_diff()

# =========================================================
# EMA
# =========================================================

df["EMA_20"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=20
).ema_indicator()

df["EMA_50"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=50
).ema_indicator()

# =========================================================
# SMA
# =========================================================

df["SMA_20"] = ta.trend.SMAIndicator(
    close=df["Close"],
    window=20
).sma_indicator()

# =========================================================
# BOLLINGER BANDS
# =========================================================

bb = ta.volatility.BollingerBands(
    close=df["Close"],
    window=20
)

df["BB_HIGH"] = bb.bollinger_hband()

df["BB_LOW"] = bb.bollinger_lband()

df["BB_WIDTH"] = bb.bollinger_wband()

# =========================================================
# ATR
# =========================================================

df["ATR"] = ta.volatility.AverageTrueRange(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14
).average_true_range()

# =========================================================
# STOCHASTIC
# =========================================================

stoch = ta.momentum.StochasticOscillator(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14
)

df["STOCH"] = stoch.stoch()

df["STOCH_SIGNAL"] = stoch.stoch_signal()

# =========================================================
# RETURNS
# =========================================================

df["Returns"] = df["Close"].pct_change()

# =========================================================
# VOLATILITY
# =========================================================

df["Volatility"] = (
    df["Returns"]
    .rolling(20)
    .std()
)

# =========================================================
# VOLUME CHANGE
# =========================================================

df["Volume_Change"] = (
    df["Volume"]
    .pct_change()
)

# =========================================================
# TREND
# =========================================================

df["Trend"] = np.where(
    df["Close"].shift(-1) > df["Close"],
    1,
    0
)

# =========================================================
# REMOVE NaN
# =========================================================

df = df.dropna()

# =========================================================
# FEATURES
# =========================================================

features = [
    "Close",
    "Volume",
    "RSI",
    "MACD",
    "MACD_SIGNAL",
    "MACD_DIFF",
    "EMA_20",
    "EMA_50",
    "SMA_20",
    "BB_HIGH",
    "BB_LOW",
    "BB_WIDTH",
    "ATR",
    "STOCH",
    "STOCH_SIGNAL",
    "Returns",
    "Volatility",
    "Volume_Change",
    "Trend"
]

latest = df[features].tail(1)

# =========================================================
# PREDICTION
# =========================================================

prediction = model.predict(latest)[0]

probability = model.predict_proba(latest)[0]

confidence = round(
    max(probability) * 100,
    2
)

# =========================================================
# LIVE PRICE
# =========================================================

live_price = round(
    df["Close"].iloc[-1],
    2
)

previous_price = round(
    df["Close"].iloc[-2],
    2
)

price_change = round(
    live_price - previous_price,
    2
)

change_percent = round(
    (price_change / previous_price) * 100,
    2
)

# =========================================================
# LIVE PRICE SECTION
# =========================================================

st.subheader("Live BTC Price")

st.metric(
    label="BTC/USD",
    value=f"${live_price}",
    delta=f"{price_change} ({change_percent}%)"
)

# =========================================================
# AI PREDICTION CARD
# =========================================================

if prediction == 1:

    st.markdown(f"""
    <div class="card green-card">
        <div class="indicator-title">
            AI Prediction : BTC may go UP
        </div>
        <br>
        Confidence : {confidence}%
    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown(f"""
    <div class="card red-card">
        <div class="indicator-title">
            AI Prediction : BTC may go DOWN
        </div>
        <br>
        Confidence : {confidence}%
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# RSI CARD
# =========================================================

latest_rsi = round(
    df["RSI"].iloc[-1],
    2
)

if latest_rsi > 70:

    rsi_text = f"RSI Overbought : {latest_rsi}"

    rsi_class = "red-card"

elif latest_rsi < 30:

    rsi_text = f"RSI Oversold : {latest_rsi}"

    rsi_class = "green-card"

else:

    rsi_text = f"RSI Neutral : {latest_rsi}"

    rsi_class = "blue-card"

st.markdown(f"""
<div class="card {rsi_class}">
    <div class="indicator-title">
        {rsi_text}
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# MACD CARD
# =========================================================

macd_value = df["MACD"].iloc[-1]

signal_value = df["MACD_SIGNAL"].iloc[-1]

if macd_value > signal_value:

    st.markdown("""
    <div class="card green-card">
        <div class="indicator-title">
            MACD Bullish Crossover
        </div>
    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <div class="card red-card">
        <div class="indicator-title">
            MACD Bearish Crossover
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# TRADINGVIEW CHART
# =========================================================

st.subheader("BTC Live Candlestick Chart")

tradingview_widget = """
<!-- TradingView Widget BEGIN -->
<div class="tradingview-widget-container">
  <div id="tradingview_chart"></div>

  <script type="text/javascript"
  src="https://s3.tradingview.com/tv.js"></script>

  <script type="text/javascript">

  new TradingView.widget(
  {
    "autosize": true,
    "symbol": "BINANCE:BTCUSDT",
    "interval": "60",
    "timezone": "Asia/Kolkata",
    "theme": "dark",
    "style": "1",
    "locale": "en",
    "toolbar_bg": "#0f172a",
    "enable_publishing": false,
    "allow_symbol_change": true,
    "container_id": "tradingview_chart"
  }
  );

  </script>

</div>
<!-- TradingView Widget END -->
"""

st.components.v1.html(
    tradingview_widget,
    height=700
)

# =========================================================
# MAIN INDICATORS
# =========================================================

st.subheader("Main Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "RSI",
        round(df["RSI"].iloc[-1], 2)
    )

with col2:

    st.metric(
        "MACD",
        round(df["MACD"].iloc[-1], 2)
    )

with col3:

    st.metric(
        "ATR",
        round(df["ATR"].iloc[-1], 2)
    )

with col4:

    st.metric(
        "Volatility",
        round(df["Volatility"].iloc[-1], 4)
    )

# =========================================================
# EMA CHART
# =========================================================

st.subheader("EMA 20 vs EMA 50")

ema_fig = go.Figure()

ema_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA_20"],
        name="EMA 20"
    )
)

ema_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA_50"],
        name="EMA 50"
    )
)

ema_fig.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(
    ema_fig,
    use_container_width=True
)

# =========================================================
# RSI CHART
# =========================================================

st.subheader("RSI Chart")

rsi_fig = go.Figure()

rsi_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["RSI"],
        name="RSI"
    )
)

rsi_fig.add_hline(
    y=70,
    line_dash="dash"
)

rsi_fig.add_hline(
    y=30,
    line_dash="dash"
)

rsi_fig.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(
    rsi_fig,
    use_container_width=True
)

# =========================================================
# MACD CHART
# =========================================================

st.subheader("MACD Chart")

macd_fig = go.Figure()

macd_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MACD"],
        name="MACD"
    )
)

macd_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MACD_SIGNAL"],
        name="MACD Signal"
    )
)

macd_fig.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(
    macd_fig,
    use_container_width=True
)

# =========================================================
# LATEST DATA
# =========================================================

st.subheader("Latest BTC Data")

st.dataframe(
    df.tail(10),
    use_container_width=True
)
