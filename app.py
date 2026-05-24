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
    interval="1d"
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
# FLOAT CONVERSION
# =====================================

for col in [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]:

    df[col] = df[col].astype(float)



# =====================================
# INDICATORS
# =====================================

# RSI
df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Close"]
).rsi()



# MACD
macd = ta.trend.MACD(
    close=df["Close"]
)

df["MACD"] = macd.macd()

df["MACD_SIGNAL"] = macd.macd_signal()



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



# SMA 20
df["SMA_20"] = ta.trend.SMAIndicator(
    close=df["Close"],
    window=20
).sma_indicator()



# RETURNS
df["Returns"] = df["Close"].pct_change()



# VOLATILITY
df["Volatility"] = (
    df["High"] - df["Low"]
) / df["Close"]



# =====================================
# DROP EMPTY ROWS
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
# PREDICTION
# =====================================

prediction = model.predict(
    latest
)



# =====================================
# CURRENT PRICE
# =====================================

current_price = df["Close"].iloc[-1]

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
# CHARTS
# =====================================

st.subheader("BTC Close Price")

st.line_chart(
    df["Close"]
)



st.subheader("RSI")

st.line_chart(
    df["RSI"]
)



st.subheader("MACD")

st.line_chart(
    df[[
        "MACD",
        "MACD_SIGNAL"
    ]]
)



# =====================================
# TABLE
# =====================================

st.subheader(
    "Latest BTC Data"
)

st.dataframe(
    df.tail()
)
