import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
import plotly.express as px
import pickle
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

st_autorefresh(interval=5000, key="btc_refresh")

# =========================================================
# LOAD MODEL
# =========================================================

model = pickle.load(open("btc_model.pkl", "rb"))

# =========================================================
# TITLE
# =========================================================

st.title("BTC AI Prediction Dashboard")

# =========================================================
# DOWNLOAD BTC DATA
# =========================================================

@st.cache_data(ttl=60)
def load_data():

    df = yf.download(
        "BTC-USD",
        period="60d",
        interval="1h"
    )

    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    df = df.dropna()

    return df

df = load_data()

# =========================================================
# TECHNICAL INDICATORS
# =========================================================

# RSI
df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Close"],
    window=14
).rsi()

# MACD
macd = ta.trend.MACD(close=df["Close"])

df["MACD"] = macd.macd()
df["MACD_SIGNAL"] = macd.macd_signal()
df["MACD_DIFF"] = macd.macd_diff()

# EMA
df["EMA_20"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=20
).ema_indicator()

df["EMA_50"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=50
).ema_indicator()

# SMA
df["SMA_20"] = ta.trend.SMAIndicator(
    close=df["Close"],
    window=20
).sma_indicator()

# BOLLINGER BANDS
bb = ta.volatility.BollingerBands(
    close=df["Close"],
    window=20,
    window_dev=2
)

df["BB_HIGH"] = bb.bollinger_hband()
df["BB_LOW"] = bb.bollinger_lband()
df["BB_WIDTH"] = bb.bollinger_wband()

# ATR
df["ATR"] = ta.volatility.AverageTrueRange(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14
).average_true_range()

# STOCHASTIC
stoch = ta.momentum.StochasticOscillator(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14,
    smooth_window=3
)

df["STOCH"] = stoch.stoch()
df["STOCH_SIGNAL"] = stoch.stoch_signal()

# RETURNS
df["Returns"] = df["Close"].pct_change()

# VOLATILITY
df["Volatility"] = df["Returns"].rolling(24).std()

# VOLUME CHANGE
df["Volume_Change"] = df["Volume"].pct_change()

# TREND
df["Trend"] = np.where(
    df["EMA_20"] > df["EMA_50"],
    1,
    0
)

# =========================================================
# CLEAN DATA
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

# =========================================================
# LATEST DATA
# =========================================================

latest = df[features].tail(1)

# =========================================================
# AI PREDICTION
# =========================================================

prediction = model.predict(latest)[0]

try:
    probability = model.predict_proba(latest)[0]

    up_probability = probability[1] * 100
    down_probability = probability[0] * 100

except:
    up_probability = 0
    down_probability = 0

# =========================================================
# LIVE PRICE
# =========================================================

current_price = df["Close"].iloc[-1]
previous_price = df["Close"].iloc[-2]

price_change = current_price - previous_price

# =========================================================
# PRICE SECTION
# =========================================================

st.subheader("Live BTC Price")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        label="BTC/USD",
        value=f"${current_price:,.2f}",
        delta=f"{price_change:.2f}"
    )

with col2:

    st.metric(
        label="AI Bullish Probability",
        value=f"{up_probability:.2f}%"
    )

# =========================================================
# AI SIGNAL
# =========================================================

if prediction == 1:

    st.success(
        f"AI Prediction: BTC may go UP | Confidence: {up_probability:.2f}%"
    )

else:

    st.error(
        f"AI Prediction: BTC may go DOWN | Confidence: {down_probability:.2f}%"
    )

# =========================================================
# RSI SIGNAL
# =========================================================

latest_rsi = df["RSI"].iloc[-1]

if latest_rsi > 70:

    st.warning(f"RSI Overbought : {latest_rsi:.2f}")

elif latest_rsi < 30:

    st.success(f"RSI Oversold : {latest_rsi:.2f}")

else:

    st.info(f"RSI Neutral : {latest_rsi:.2f}")

# =========================================================
# MACD SIGNAL
# =========================================================

latest_macd = df["MACD"].iloc[-1]
latest_signal = df["MACD_SIGNAL"].iloc[-1]

if latest_macd > latest_signal:

    st.success("MACD Bullish Crossover")

else:

    st.error("MACD Bearish Crossover")

# =========================================================
# CANDLESTICK CHART
# =========================================================

st.subheader("BTC Live Candlestick Chart")

chart_df = df.tail(200)

fig = go.Figure(
    data=[
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="BTC"
        )
    ]
)

fig.update_layout(
    template="plotly_dark",
    height=700,
    xaxis_rangeslider_visible=False,
    title="BTC/USD Live Candlestick"
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# RSI CHART
# =========================================================

st.subheader("RSI")

rsi_df = df.tail(200)

fig_rsi = go.Figure()

fig_rsi.add_trace(
    go.Scatter(
        x=rsi_df.index,
        y=rsi_df["RSI"],
        mode="lines",
        name="RSI"
    )
)

fig_rsi.add_hline(y=70)
fig_rsi.add_hline(y=30)

fig_rsi.update_layout(
    template="plotly_dark",
    height=400
)

st.plotly_chart(fig_rsi, use_container_width=True)

# =========================================================
# MACD CHART
# =========================================================

st.subheader("MACD")

macd_df = df.tail(200)

fig_macd = go.Figure()

fig_macd.add_trace(
    go.Scatter(
        x=macd_df.index,
        y=macd_df["MACD"],
        mode="lines",
        name="MACD"
    )
)

fig_macd.add_trace(
    go.Scatter(
        x=macd_df.index,
        y=macd_df["MACD_SIGNAL"],
        mode="lines",
        name="MACD SIGNAL"
    )
)

fig_macd.update_layout(
    template="plotly_dark",
    height=400
)

st.plotly_chart(fig_macd, use_container_width=True)

# =========================================================
# EMA CHART
# =========================================================

st.subheader("EMA 20 vs EMA 50")

ema_df = df.tail(200)

fig_ema = go.Figure()

fig_ema.add_trace(
    go.Scatter(
        x=ema_df.index,
        y=ema_df["Close"],
        mode="lines",
        name="Close"
    )
)

fig_ema.add_trace(
    go.Scatter(
        x=ema_df.index,
        y=ema_df["EMA_20"],
        mode="lines",
        name="EMA 20"
    )
)

fig_ema.add_trace(
    go.Scatter(
        x=ema_df.index,
        y=ema_df["EMA_50"],
        mode="lines",
        name="EMA 50"
    )
)

fig_ema.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(fig_ema, use_container_width=True)

# =========================================================
# DATA TABLE
# =========================================================

st.subheader("Latest BTC Data")

st.dataframe(
    df.tail(10),
    use_container_width=True
)
