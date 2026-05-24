import streamlit as st
import pandas as pd
import requests
import ta
import pickle



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
    "limit": 200
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

df = pd.DataFrame(data)



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
# INDICATORS
# =========================================

df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Close"]
).rsi()



macd = ta.trend.MACD(
    close=df["Close"]
)

df["MACD"] = macd.macd()

df["MACD_SIGNAL"] = macd.macd_signal()



df["EMA_20"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=20
).ema_indicator()



df["EMA_50"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=50
).ema_indicator()



df["SMA_20"] = ta.trend.SMAIndicator(
    close=df["Close"],
    window=20
).sma_indicator()



df["Returns"] = df["Close"].pct_change()



df["Volatility"] = (
    df["High"] - df["Low"]
) / df["Close"]



# =========================================
# CLEAN DATA
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
# PREDICTION
# =========================================

prediction = model.predict(
    latest
)



# =========================================
# SHOW RESULT
# =========================================

if prediction[0] == 1:
    st.success("BTC Prediction: UP")
else:
    st.error("BTC Prediction: DOWN")



# =========================================
# SHOW PRICE CHART
# =========================================

st.subheader("BTC Close Price")

st.line_chart(df["Close"])



# =========================================
# SHOW RSI
# =========================================

st.subheader("RSI")

st.line_chart(df["RSI"])



# =========================================
# SHOW DATA
# =========================================

st.subheader("Latest BTC Data")

st.dataframe(df.tail())
