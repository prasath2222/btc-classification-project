import pickle
import pandas as pd
import requests
import ta



# =========================================
# LOAD MODEL
# =========================================

model = pickle.load(
    open("btc_model.pkl", "rb")
)



# =========================================
# GET LATEST BTC DATA
# =========================================

url = "https://api.binance.com/api/v3/klines"

params = {
    "symbol": "BTCUSDT",
    "interval": "1d",
    "limit": 100
}



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



# EMA
df["EMA_20"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=20
).ema_indicator()



# SMA
df["SMA_20"] = ta.trend.SMAIndicator(
    close=df["Close"],
    window=20
).sma_indicator()



# RETURNS
df["Returns"] = df["Close"].pct_change()



# =========================================
# REMOVE EMPTY ROWS
# =========================================

df = df.dropna()



# =========================================
# LATEST ROW
# =========================================

latest = df[[
    "Close",
    "Volume",
    "RSI",
    "MACD",
    "EMA_20",
    "SMA_20",
    "Returns"
]].tail(1)



# =========================================
# PREDICT
# =========================================

prediction = model.predict(latest)



# =========================================
# OUTPUT
# =========================================

if prediction[0] == 1:
    print("Prediction: BTC may go UP")
else:
    print("Prediction: BTC may go DOWN")import pickle
import numpy as np



# LOAD MODEL
model = pickle.load(
    open("btc_model.pkl", "rb")
)



# NEW DATA
# Close, Volume
new_data = np.array([
    [65000, 30000000000]
])



# PREDICT
prediction = model.predict(new_data)



# OUTPUT
if prediction[0] == 1:
    print("BTC may go UP")
else:
    print("BTC may go DOWN")
