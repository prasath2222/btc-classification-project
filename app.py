import streamlit as st
from streamlit_autorefresh import st_autorefresh

import pandas as pd
import numpy as np
import ta
import pickle
import yfinance as yf

import websocket
import json
import threading



# =====================================
# AUTO REFRESH
# =====================================

st_autorefresh(
    interval=60000,
    key="btc_refresh"
)



# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="BTC AI Dashboard",
    layout="wide"
)



# =====================================
# TITLE
# =====================================

st.title("BTC AI Prediction Dashboard")



# =====================================
# LOAD MODEL
# =====================================

model = pickle.load(
    open("btc_model.pkl", "rb")
)



# =====================================
# LIVE PRICE PLACEHOLDER
# =====================================

live_price_placeholder = st.empty()



# =====================================
# SESSION STATE
# =====================================

if "live_price" not in st.session_state:

    st.session_state.live_price = 0



# =====================================
# WEBSOCKET MESSAGE
# =====================================

def on_message(ws, message):

    data = json.loads(message)

    st.session_state.live_price = float(
        data["p"]
    )



# =====================================
# START WEBSOCKET
# =====================================

def start_websocket():

    ws = websocket.WebSocketApp(

        "wss://stream.binance.com:9443/ws/btcusdt@trade",

        on_message=on_message

    )

    ws.run_forever()



# =====================================
# THREAD
# =====================================

threading.Thread(

    target=start_websocket,
    daemon=True

).start()



# =====================================
# SHOW LIVE PRICE
# =====================================

live_price_placeholder.metric(

    "Live BTC Price",

    f"${st.session_state.live_price:,.2f}"

)



# =====================================
# DOWNLOAD BTC DATA
# =====================================

df = yf.download(

    "BTC-USD",

    period="5y",

    interval="1d",

    auto_adjust=True

)



# =====================================
# FIX MULTIINDEX
# =====================================

if isinstance(df.columns, pd.MultiIndex):

    df.columns = df.columns.get_level_values(0)



# =====================================
# RESET INDEX
# =====================================

df.reset_index(inplace=True)



# =====================================
# KEEP COLUMNS
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

for col in df.columns:

    df[col] = pd.to_numeric(

        df[col],
        errors="coerce"

    )



# =====================================
# RSI
# =====================================

df["RSI"] = ta.momentum.RSIIndicator(

    close=df["Close"],
    window=14

).rsi()



# =====================================
# MACD
# =====================================

macd = ta.trend.MACD(

    close=df["Close"]

)

df["MACD"] = macd.macd()

df["MACD_SIGNAL"] = macd.macd_signal()

df["MACD_DIFF"] = macd.macd_diff()



# =====================================
# EMA
# =====================================

df["EMA_20"] = ta.trend.EMAIndicator(

    close=df["Close"],
    window=20

).ema_indicator()



df["EMA_50"] = ta.trend.EMAIndicator(

    close=df["Close"],
    window=50

).ema_indicator()



# =====================================
# SMA
# =====================================

df["SMA_20"] = ta.trend.SMAIndicator(

    close=df["Close"],
    window=20

).sma_indicator()



# =====================================
# BOLLINGER BANDS
# =====================================

bb = ta.volatility.BollingerBands(

    close=df["Close"],
    window=20

)

df["BB_HIGH"] = bb.bollinger_hband()

df["BB_LOW"] = bb.bollinger_lband()

df["BB_WIDTH"] = bb.bollinger_wband()



# =====================================
# ATR
# =====================================

df["ATR"] = ta.volatility.AverageTrueRange(

    high=df["High"],
    low=df["Low"],
    close=df["Close"]

).average_true_range()



# =====================================
# STOCHASTIC
# =====================================

stoch = ta.momentum.StochasticOscillator(

    high=df["High"],
    low=df["Low"],
    close=df["Close"]

)

df["STOCH"] = stoch.stoch()

df["STOCH_SIGNAL"] = stoch.stoch_signal()



# =====================================
# RETURNS
# =====================================

df["Returns"] = df["Close"].pct_change()



# =====================================
# VOLATILITY
# =====================================

df["Volatility"] = (

    df["High"] - df["Low"]

) / df["Close"]



# =====================================
# VOLUME CHANGE
# =====================================

df["Volume_Change"] = (

    df["Volume"].pct_change()

)



# =====================================
# DROP NAN
# =====================================

df.dropna(inplace=True)



# =====================================
# FEATURES
# =====================================

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
    "Volume_Change"

]



latest = df[features].tail(1)



# =====================================
# PREDICTION
# =====================================

prediction = model.predict(

    latest

)[0]



# =====================================
# PROBABILITY
# =====================================

probability = model.predict_proba(

    latest

)[0]



up_prob = probability[1] * 100

down_prob = probability[0] * 100



# =====================================
# SIGNAL
# =====================================

st.subheader("AI Market Signal")



if prediction == 1:

    st.success(

        f"BTC may go UP\n\n"
        f"UP Probability: {up_prob:.2f}%"

    )

else:

    st.error(

        f"BTC may go DOWN\n\n"
        f"DOWN Probability: {down_prob:.2f}%"

    )



# =====================================
# CURRENT PRICE
# =====================================

current_price = df["Close"].iloc[-1]



st.metric(

    "Daily Close Price",

    f"${current_price:,.2f}"

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
# DATA TABLE
# =====================================

st.subheader("Latest BTC Data")

st.dataframe(

    df.tail()

)
