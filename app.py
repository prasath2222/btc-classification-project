import streamlit as st
from streamlit_autorefresh import st_autorefresh

import pandas as pd
import ta
import pickle
import yfinance as yf



# =====================================
# AUTO REFRESH
# =====================================

st_autorefresh(
    interval=300000,
    key="btc_refresh"
)



# =====================================
# PAGE TITLE
# =====================================

st.title("BTC AI Prediction Dashboard")



# =====================================
# LOAD MODEL
# =====================================

model = pickle.load(
    open("btc_model.pkl", "rb")
)



# =====================================
# DOWNLOAD BTC DATA
# =====================================

df = yf.download(
    "BTC-USD",
    period="300d",
    interval="1d",
    auto_adjust=True
)



# =====================================
# FIX MULTI-INDEX COLUMNS
# =====================================

if isinstance(df.columns, pd.MultiIndex):

    df.columns = (
        df.columns.get_level_values(0)
    )



# =====================================
# RESET INDEX
# =====================================

df.reset_index(inplace=True)



# =====================================
# KEEP NEEDED COLUMNS
# =====================================

df = df[[
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]]



# =====================================
# CONVERT TO FLOAT
# =====================================

for col in [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )



# =====================================
# FORCE 1D SERIES
# =====================================

close_series = (
    df["Close"].squeeze()
)



# =====================================
# RSI
# =====================================

df["RSI"] = ta.momentum.RSIIndicator(
    close=close_series
).rsi()



# =====================================
# MACD
# =====================================

macd = ta.trend.MACD(
    close=close_series
)

df["MACD"] = macd.macd()

df["MACD_SIGNAL"] = (
    macd.macd_signal()
)



# =====================================
# EMA 20
# =====================================

df["EMA_20"] = ta.trend.EMAIndicator(
    close=close_series,
    window=20
).ema_indicator()



# =====================================
# EMA 50
# =====================================

df["EMA_50"] = ta.trend.EMAIndicator(
    close=close_series,
    window=50
).ema_indicator()



# =====================================
# SMA 20
# =====================================

df["SMA_20"] = ta.trend.SMAIndicator(
    close=close_series,
    window=20
).sma_indicator()



# =====================================
# RETURNS
# =====================================

df["Returns"] = (
    close_series.pct_change()
)



# =====================================
# VOLATILITY
# =====================================

df["Volatility"] = (
    (df["High"] - df["Low"])
    / close_series
)



# =====================================
# REMOVE EMPTY ROWS
# =====================================

df.dropna(inplace=True)



# =====================================
# FEATURES
# =====================================

latest = df[[
    "Close",
    "Volume",
    "RSI",
    "MACD",
    "MACD_SIGNAL",
    "EMA_20",
    "EMA_50",
    "SMA_20",
    "Returns",
    "Volatility"
]].tail(1)



# =====================================
# MODEL PREDICTION
# =====================================

prediction = model.predict(
    latest
)



# =====================================
# CURRENT BTC PRICE
# =====================================

current_price = (
    close_series.iloc[-1]
)

st.metric(
    "Current BTC Price",
    f"${current_price:,.2f}"
)



# =====================================
# AI RESULT
# =====================================

if prediction[0] == 1:

    st.success(
        "AI Prediction: BTC may go UP"
    )

else:

    st.error(
        "AI Prediction: BTC may go DOWN"
    )



# =====================================
# BTC PRICE CHART
# =====================================

st.subheader(
    "BTC Close Price"
)

st.line_chart(
    close_series
)



# =====================================
# RSI CHART
# =====================================

st.subheader("RSI")

st.line_chart(
    df["RSI"]
)



# =====================================
# MACD CHART
# =====================================

st.subheader("MACD")

st.line_chart(
    df[[
        "MACD",
        "MACD_SIGNAL"
    ]]
)



# =====================================
# LATEST DATA TABLE
# =====================================

st.subheader(
    "Latest BTC Data"
)

st.dataframe(
    df.tail()
)
