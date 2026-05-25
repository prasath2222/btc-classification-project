# =========================================================
# BTC AI DASHBOARD APP.PY
# =========================================================

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
# LOAD MODEL
# =========================================================

model = pickle.load(
    open("btc_model.pkl", "rb")
)

# =========================================================
# TITLE
# =========================================================

st.title("BTC AI Prediction Dashboard")

# =========================================================
# DOWNLOAD BTC DATA
# =========================================================

df = yf.download(
    "BTC-USD",
    period="730d",
    interval="1h"
)

# =========================================================
# FIX MULTI INDEX
# =========================================================

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# =========================================================
# INDICATORS
# =========================================================

# RSI
df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Close"],
    window=14
).rsi()

# MACD
macd = ta.trend.MACD(
    close=df["Close"]
)

df["MACD"] = macd.macd()

df["MACD_SIGNAL"] = (
    macd.macd_signal()
)

# EMA 20
df["EMA_20"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=20
).ema_indicator()

# EMA 50
df["EMA_50"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=50
).ema_indicator()

# ATR
df["ATR"] = ta.volatility.AverageTrueRange(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14
).average_true_range()

# RETURNS
df["Returns"] = (
    df["Close"].pct_change()
)

# VOLATILITY
df["Volatility"] = (
    df["Returns"].rolling(20).std()
)

# TREND
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
# FORCE NUMERIC
# =========================================================

for col in features:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df.dropna(inplace=True)

# =========================================================
# LATEST DATA
# =========================================================

latest = df[features].iloc[-1:]

# =========================================================
# PREDICTION
# =========================================================

prediction = model.predict(
    latest
)[0]

# =========================================================
# LIVE PRICE
# =========================================================

current_price = (
    float(df["Close"].iloc[-1])
)

previous_price = (
    float(df["Close"].iloc[-2])
)

price_change = (
    current_price - previous_price
)

# =========================================================
# LIVE PRICE DISPLAY
# =========================================================

st.subheader("Live BTC Price")

st.metric(
    label="BTC/USD",
    value=f"${current_price:,.2f}",
    delta=f"{price_change:.2f}"
)

# =========================================================
# AI PREDICTION
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

latest_rsi = (
    float(df["RSI"].iloc[-1])
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

latest_macd = (
    float(df["MACD"].iloc[-1])
)

latest_signal = (
    float(df["MACD_SIGNAL"].iloc[-1])
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
    height=500,
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# RSI CHART
# =========================================================

st.subheader("RSI")

fig_rsi = go.Figure()

fig_rsi.add_trace(
    go.Scatter(
        y=df["RSI"],
        mode="lines",
        name="RSI"
    )
)

fig_rsi.update_layout(
    height=400,
    template="plotly_dark"
)

st.plotly_chart(
    fig_rsi,
    use_container_width=True
)

# =========================================================
# MACD CHART
# =========================================================

st.subheader("MACD")

fig_macd = go.Figure()

fig_macd.add_trace(
    go.Scatter(
        y=df["MACD"],
        mode="lines",
        name="MACD"
    )
)

fig_macd.add_trace(
    go.Scatter(
        y=df["MACD_SIGNAL"],
        mode="lines",
        name="MACD_SIGNAL"
    )
)

fig_macd.update_layout(
    height=400,
    template="plotly_dark"
)

st.plotly_chart(
    fig_macd,
    use_container_width=True
)

# =========================================================
# EMA CHART
# =========================================================

st.subheader("EMA 20 vs EMA 50")

fig_ema = go.Figure()

fig_ema.add_trace(
    go.Scatter(
        y=df["EMA_20"],
        mode="lines",
        name="EMA 20"
    )
)

fig_ema.add_trace(
    go.Scatter(
        y=df["EMA_50"],
        mode="lines",
        name="EMA 50"
    )
)

fig_ema.update_layout(
    height=400,
    template="plotly_dark"
)

st.plotly_chart(
    fig_ema,
    use_container_width=True
)

# =========================================================
# LATEST DATA TABLE
# =========================================================

st.subheader("Latest BTC Data")

st.dataframe(
    df.tail(10)
)
