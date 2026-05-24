import streamlit as st
from streamlit_autorefresh import st_autorefresh

import pandas as pd
import requests
import ta
import pickle



# =========================================
# AUTO REFRESH
# =========================================

st_autorefresh(
    interval=60000,
    key="btc_refresh"
)



# =========================================
# LOAD MODEL
# =========================================

model = pickle.load(
    open("btc_model.pkl", "rb")
)



# =========================================
# PAGE TITLE
# =========================================

st.title("BTC AI Prediction Dashboard")



# =========================================
# BINANCE API
# =========================================

url = "https://api.binance.com/api/v3/klines"

params = {
    "symbol": "BTCUSDT",
    "interval": "1d",
    "limit": 300
}



# =========================================
# DOWNLOAD DATA
# =========================================

response = requests.get(
    url,
    params=params
)

data = response.json()



# =========================================
# CREATE DATAFRAME
# =========================================

df = pd.DataFrame([data])



# =========================================
# SELECT COLUMNS
# =========================================

df = df[[1,2,3,4,5]]

df.columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]



# =========================================
# CONVERT TO FLOAT
# =========================================

for col in df.columns:
    df[col] = df[col].astype(float)



# =========================================
# TECHNICAL INDICATORS
# =========================================

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



# =========================================
# REMOVE EMPTY ROWS
# =========================================

df = df.dropna()



# =========================================
# FEATURES
# =========================================

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



# =========================================
# PREDICT
# =========================================

prediction = model.predict(
    latest
)



# =========================================
# CURRENT BTC PRICE
# =========================================

current_price = df["Close"].iloc[-1]

st.metric(
    label="Current BTC Price",
    value=f"${current_price:,.2f}"
)



# =========================================
# AI PREDICTION
# =========================================

if prediction[0] == 1:

    st.success(
        "AI Prediction: BTC may go UP"
    )

else:

    st.error(
        "AI Prediction: BTC may go DOWN"
    )



# =========================================
# PRICE CHART
# =========================================

st.subheader("BTC Close Price")

st.line_chart(
    df["Close"]
)



# =========================================
# RSI CHART
# =========================================

st.subheader("RSI")

st.line_chart(
    df["RSI"]
)



# =========================================
# MACD CHART
# =========================================

st.subheader("MACD")

st.line_chart(
    df[[
        "MACD",
        "MACD_SIGNAL"
    ]]
)



# =========================================
# SHOW LATEST DATA
# =========================================

st.subheader(
    "Latest BTC Data"
)

st.dataframe(
    df.tail()
)
