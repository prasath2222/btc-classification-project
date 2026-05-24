from fastapi import FastAPI

import pickle
import pandas as pd
import ta
import yfinance as yf



# =========================================
# FASTAPI APP
# =========================================

app = FastAPI()



# =========================================
# LOAD MODEL
# =========================================

model = pickle.load(
    open("btc_model.pkl", "rb")
)



# =========================================
# HOME
# =========================================

@app.get("/")
def home():

    return {
        "message": "BTC AI API Running"
    }



# =========================================
# PREDICT ENDPOINT
# =========================================

@app.get("/predict")
def predict():



    # =========================================
    # DOWNLOAD BTC DATA
    # =========================================

    df = yf.download(
        "BTC-USD",
        period="300d",
        interval="1d"
    )



    # =========================================
    # RESET INDEX
    # =========================================

    df.reset_index(inplace=True)



    # =========================================
    # KEEP NEEDED COLUMNS
    # =========================================

    df = df[[
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]]



    # =========================================
    # FLOAT CONVERSION
    # =========================================

    for col in df.columns:

        df[col] = df[col].astype(float)



    # =========================================
    # RSI
    # =========================================

    df["RSI"] = ta.momentum.RSIIndicator(
        close=df["Close"]
    ).rsi()



    # =========================================
    # MACD
    # =========================================

    macd = ta.trend.MACD(
        close=df["Close"]
    )

    df["MACD"] = macd.macd()

    df["MACD_SIGNAL"] = macd.macd_signal()



    # =========================================
    # EMA
    # =========================================

    df["EMA_20"] = ta.trend.EMAIndicator(
        close=df["Close"],
        window=20
    ).ema_indicator()



    df["EMA_50"] = ta.trend.EMAIndicator(
        close=df["Close"],
        window=50
    ).ema_indicator()



    # =========================================
    # SMA
    # =========================================

    df["SMA_20"] = ta.trend.SMAIndicator(
        close=df["Close"],
        window=20
    ).sma_indicator()



    # =========================================
    # RETURNS
    # =========================================

    df["Returns"] = (
        df["Close"].pct_change()
    )



    # =========================================
    # VOLATILITY
    # =========================================

    df["Volatility"] = (
        df["High"] - df["Low"]
    ) / df["Close"]



    # =========================================
    # CLEAN NAN VALUES
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
    # SIGNAL
    # =========================================

    if prediction[0] == 1:

        signal = "UP"

    else:

        signal = "DOWN"



    # =========================================
    # RETURN JSON
    # =========================================

    return {

        "btc_price":
        float(df["Close"].iloc[-1]),

        "prediction":
        signal,

        "rsi":
        float(df["RSI"].iloc[-1]),

        "macd":
        float(df["MACD"].iloc[-1])

    }
