import streamlit as st
from streamlit_autorefresh import st_autorefresh

import pandas as pd
import requests
import ta
import pickle



# =====================================
# AUTO REFRESH
# =====================================

st_autorefresh(
    interval=60000,
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
# COINGECKO API
# =====================================

url = (
    "https://api.coingecko.com/api/v3/"
    "coins/bitcoin/market_chart"
)

params = {
    "vs_currency": "usd",
    "days": 300,
    "interval": "daily"
}



# =====================================
# GET DATA
# =====================================

response = requests.get(
    url,
    params=params
)

data = response.json()



# =====================================
# CHECK API
# =====================================

if "prices" not in data:

    st.error("API Error")
    st.write(data)
    st.stop()



# =====================================
# CREATE DATAFRAME
# =====================================

prices = data["prices"]

df = pd.DataFrame(
    prices,
    columns=[
        "Time",
        "Close"
    ]
)



# =====================================
# FLOAT CONVERSION
# =====================================

df["Close"] = df["Close"].astype(float)



# =====================================
# CREATE DUMMY COLUMNS
# =====================================

df["Open"] = df["Close"]

df["High"] = (
    df["Close"] * 1.01
)

df["Low"] = (
    df["Close"] * 0.99
)

df["Volume"] = 1000



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
