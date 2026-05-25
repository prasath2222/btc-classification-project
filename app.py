import streamlit as st
from streamlit_autorefresh import st_autorefresh

import pandas as pd
import ta
import pickle
import yfinance as yf



# =========================================
# AUTO REFRESH
# =========================================

st_autorefresh(
    interval=60000,
    key="btc_refresh"
)



# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="BTC AI Dashboard",
    layout="wide"
)



# =========================================
# TITLE
# =========================================

st.title("BTC AI Prediction Dashboard")



# =========================================
# LOAD MODEL
# =========================================

model = pickle.load(
    open("btc_model.pkl", "rb")
)



# =========================================
# DOWNLOAD BTC DATA
# =========================================

df = yf.download(
    "BTC-USD",
    period="300d",
    interval="1h",
    auto_adjust=True
)



# =========================================
# CHECK EMPTY
# =========================================

if df.empty:

    st.error("Failed to load BTC data")
    st.stop()



# =========================================
# FIX MULTIINDEX
# =========================================

if isinstance(df.columns, pd.MultiIndex):

    df.columns = df.columns.get_level_values(0)



# =========================================
# RESET INDEX
# =========================================

df.reset_index(inplace=True)



# =========================================
# KEEP COLUMNS
# =========================================

df = df[[
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]].copy()



# =========================================
# CONVERT FLOAT
# =========================================

for col in df.columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )



# =========================================
# REMOVE BAD ROWS
# =========================================

df.dropna(inplace=True)



# =========================================
# CHECK DATA SIZE
# =========================================

if len(df) < 50:

    st.error("Not enough BTC data")
    st.stop()



# =========================================
# INDICATORS
# =========================================

close = df["Close"].squeeze()

high = df["High"].squeeze()

low = df["Low"].squeeze()

volume = df["Volume"].squeeze()



# =========================================
# RSI
# =========================================

df["RSI"] = ta.momentum.RSIIndicator(
    close=close,
    window=14
).rsi()



# =========================================
# MACD
# =========================================

macd = ta.trend.MACD(
    close=close
)

df["MACD"] = macd.macd()

df["MACD_SIGNAL"] = macd.macd_signal()



# =========================================
# EMA
# =========================================

df["EMA_20"] = ta.trend.EMAIndicator(
    close=close,
    window=20
).ema_indicator()



df["EMA_50"] = ta.trend.EMAIndicator(
    close=close,
    window=50
).ema_indicator()



# =========================================
# SMA
# =========================================

df["SMA_20"] = ta.trend.SMAIndicator(
    close=close,
    window=20
).sma_indicator()



# =========================================
# RETURNS
# =========================================

df["Returns"] = close.pct_change()



# =========================================
# VOLATILITY
# =========================================

df["Volatility"] = (
    high - low
) / close



# =========================================
# ATR
# =========================================

df["ATR"] = ta.volatility.AverageTrueRange(
    high=high,
    low=low,
    close=close,
    window=14
).average_true_range()



# =========================================
# BOLLINGER BANDS
# =========================================

bb = ta.volatility.BollingerBands(
    close=close,
    window=20
)

df["BB_HIGH"] = bb.bollinger_hband()

df["BB_LOW"] = bb.bollinger_lband()



# =========================================
# ADX
# =========================================

adx = ta.trend.ADXIndicator(
    high=high,
    low=low,
    close=close,
    window=14
)

df["ADX"] = adx.adx()



# =========================================
# CLEAN FINAL NAN
# =========================================

df = df.dropna().copy()



# =========================================
# FINAL CHECK
# =========================================

if df.empty:

    st.error("Indicators failed")
    st.stop()



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
)[0]



# =========================================
# LIVE PRICE
# =========================================

live_price = float(
    df["Close"].iloc[-1]
)



# =========================================
# PRICE CHANGE
# =========================================

previous_price = float(
    df["Close"].iloc[-2]
)

change = live_price - previous_price



# =========================================
# METRICS
# =========================================

st.metric(
    "Live BTC Price",
    f"${live_price:,.2f}",
    f"{change:,.2f}"
)



# =========================================
# SIGNAL
# =========================================

if prediction == 1:

    st.success(
        "AI Prediction: BTC may go UP"
    )

else:

    st.error(
        "AI Prediction: BTC may go DOWN"
    )



# =========================================
# RSI STATUS
# =========================================

latest_rsi = float(
    df["RSI"].iloc[-1]
)

if latest_rsi > 70:

    st.warning(
        f"RSI Overbought: {latest_rsi:.2f}"
    )

elif latest_rsi < 30:

    st.warning(
        f"RSI Oversold: {latest_rsi:.2f}"
    )

else:

    st.info(
        f"RSI Neutral: {latest_rsi:.2f}"
    )



# =========================================
# MACD STATUS
# =========================================

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



# =========================================
# CHARTS
# =========================================

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



st.subheader("EMA 20 vs EMA 50")

st.line_chart(
    df[[
        "EMA_20",
        "EMA_50"
    ]]
)



st.subheader("Bollinger Bands")

st.line_chart(
    df[[
        "BB_HIGH",
        "Close",
        "BB_LOW"
    ]]
)



# =========================================
# LAST DATA
# =========================================

st.subheader(
    "Latest BTC Data"
)

st.dataframe(
    df.tail(10)
)
