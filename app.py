import streamlit as st
from streamlit_autorefresh import st_autorefresh

import pandas as pd
import numpy as np
import yfinance as yf
import ta
import pickle

import plotly.graph_objects as go

# =========================================================
# AUTO REFRESH
# =========================================================

st_autorefresh(
    interval=300000,
    key="btc_refresh"
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="BTC AI Dashboard",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("BTC AI Prediction Dashboard")

# =========================================================
# LOAD MODEL
# =========================================================

model = pickle.load(
    open("btc_model.pkl", "rb")
)

# =========================================================
# DOWNLOAD DATA
# =========================================================

df = yf.download(
    "BTC-USD",
    period="730d",
    interval="1h",
    auto_adjust=True
)

# =========================================================
# FIX EMPTY DATA
# =========================================================

if df.empty:
    st.error("Failed to download BTC data")
    st.stop()

# =========================================================
# FIX MULTI INDEX
# =========================================================

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_cols = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

for col in required_cols:

    if col not in df.columns:
        st.error(f"Missing column: {col}")
        st.stop()

# =========================================================
# FORCE NUMERIC
# =========================================================

for col in required_cols:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# =========================================================
# DROP NAN
# =========================================================

df.dropna(inplace=True)

# =========================================================
# CHECK DATA SIZE
# =========================================================

if len(df) < 100:
    st.error("Not enough BTC data")
    st.stop()

# =========================================================
# INDICATORS
# =========================================================

df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Close"],
    window=14
).rsi()

macd = ta.trend.MACD(
    close=df["Close"]
)

df["MACD"] = macd.macd()

df["MACD_SIGNAL"] = (
    macd.macd_signal()
)

df["EMA_20"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=20
).ema_indicator()

df["EMA_50"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=50
).ema_indicator()

df["ATR"] = ta.volatility.AverageTrueRange(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14
).average_true_range()

df["Returns"] = (
    df["Close"].pct_change()
)

df["Volatility"] = (
    df["Returns"].rolling(20).std()
)

df["Trend"] = (
    df["EMA_20"] > df["EMA_50"]
).astype(int)

# =========================================================
# CLEAN DATA
# =========================================================

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

df.dropna(inplace=True)

# =========================================================
# FEATURES
# =========================================================

features = [

    "RSI",

    "MACD",
    "MACD_SIGNAL",

    "EMA_20",
    "EMA_50",

    "ATR",

    "Returns",

    "Volatility",

    "Trend"
]

# =========================================================
# FINAL CHECK
# =========================================================

for col in features:

    if col not in df.columns:
        st.error(f"Missing feature: {col}")
        st.stop()

# =========================================================
# LATEST DATA
# =========================================================

latest = df[features].iloc[-1:]

# =========================================================
# PREDICT
# =========================================================

prediction = model.predict(
    latest
)[0]

# =========================================================
# PRICE
# =========================================================

current_price = float(
    df["Close"].iloc[-1]
)

previous_price = float(
    df["Close"].iloc[-2]
)

price_change = (
    current_price - previous_price
)

# =========================================================
# PRICE DISPLAY
# =========================================================

st.subheader("Live BTC Price")

st.metric(
    label="BTC/USD",
    value=f"${current_price:,.2f}",
    delta=f"{price_change:.2f}"
)

# =========================================================
# AI RESULT
# =========================================================

if prediction == 1:

    st.success(
        "AI Prediction: BTC may go UP"
    )

else:

    st.error(
        "AI Prediction: BTC may go DOWN"
    )

# =========================================================
# RSI STATUS
# =========================================================

latest_rsi = float(
    df["RSI"].iloc[-1]
)

if latest_rsi > 70:

    st.warning(
        f"RSI Overbought : {latest_rsi:.2f}"
    )

elif latest_rsi < 30:

    st.success(
        f"RSI Oversold : {latest_rsi:.2f}"
    )

else:

    st.info(
        f"RSI Neutral : {latest_rsi:.2f}"
    )

# =========================================================
# MACD STATUS
# =========================================================

latest_macd = float(
    df["MACD"].iloc[-1]
)

latest_signal = float(
    df["MACD_SIGNAL"].iloc[-1]
)

if latest_macd > latest_signal:

    st.success(
        "MACD Bullish Crossover"
    )

else:

    st.error(
        "MACD Bearish Crossover"
    )

# =========================================================
# BTC PRICE CHART
# =========================================================

st.subheader("BTC Close Price")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        y=df["Close"],
        mode="lines",
        name="BTC Price"
    )
)

fig.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# LATEST DATA
# =========================================================

st.subheader("Latest BTC Data")

st.dataframe(
    df.tail(10)
)
