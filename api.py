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
# LOAD TRAINED MODEL
# =========================================

model = pickle.load(
    open("btc_model.pkl", "rb")
)



# =========================================
# HOME ROUTE
# =========================================

@app.get("/")
def home():

    return {
        "message": "BTC AI API Running"
    }



# =========================================
# PREDICTION ROUTE
# =========================================

@app.get("/predict")
def predict():



    # =========================================
    # DOWNLOAD BTC DATA
    # =========================================

    df = yf.download(
        "BTC-USD",
        period="300d",
        interval="1d",
        auto_adjust=True
    )



    # =========================================
    # FIX MULTI-INDEX COLUMNS
    # =========================================

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = df.columns.get_level_values(0)



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
    # CONVERT TO FLOAT
    # =========================================

    for col in df.columns:

        df[col] = (
            pd.to_numeric(
                df[col],
                errors="coerce"
            )
        )



    # =========================================
    # FORCE 1D SERIES
    # =========================================

    close_series = df["Close"].squeeze()



    # =========================================
    # RSI
    # =========================================

    df["RSI"] = ta.momentum.RSIIndicator(
        close=close_series
    ).rsi()



    # =========================================
    # MACD
    # =========================================

    macd = ta.trend.MACD(
        close=close_series
    )

    df["MACD"] = macd.macd()

    df["MACD_SIGNAL"] = (
        macd.macd_signal()
    )



    # =========================================
    # EMA 20
    # =========================================

    df["EMA_20"] = ta.trend.EMAIndicator(
        close=close_series,
        window=20
    ).ema_indicator()



    # =========================================
    # EMA 50
    # =========================================

    df["EMA_50"] = ta.trend.EMAIndicator(
        close=close_series,
        window=50
    ).ema_indicator()



    # =========================================
    # SMA 20
    # =========================================

    df["SMA_20"] = ta.trend.SMAIndicator(
        close=close_series,
        window=20
    ).sma_indicator()



    # =========================================
    # RETURNS
    # =========================================

    df["Returns"] = (
        close_series.pct_change()
    )



    # =========================================
    # VOLATILITY
    # =========================================

    df["Volatility"] = (
        (df["High"] - df["Low"])
        / close_series
    )



    # =========================================
    # REMOVE EMPTY ROWS
    # =========================================

    df.dropna(inplace=True)



    # =========================================
    # MODEL FEATURES
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
    # AI PREDICTION
    # =========================================

    prediction = model.predict(
        latest
    )



    # =========================================
    # SIGNAL RESULT
    # =========================================

    if prediction[0] == 1:

        signal = "UP"

    else:

        signal = "DOWN"



    # =========================================
    # RETURN RESULT
    # =========================================

    return {

        "btc_price":
        float(close_series.iloc[-1]),

        "prediction":
        signal,

        "rsi":
        float(df["RSI"].iloc[-1]),

        "macd":
        float(df["MACD"].iloc[-1])

    }
