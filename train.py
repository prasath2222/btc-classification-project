import requests
import pandas as pd
import ta

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# =========================================
# BINANCE API
# =========================================

url = "https://api.binance.com/api/v3/klines"

params = {
    "symbol": "BTCUSDT",
    "interval": "1d",
    "limit": 1000
}



# =========================================
# DOWNLOAD DATA
# =========================================

response = requests.get(url, params=params)

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
# TARGET
# =========================================

df["target"] = (
    df["Close"].shift(-1) > df["Close"]
).astype(int)



# =========================================
# REMOVE EMPTY ROWS
# =========================================

df = df.dropna()



# =========================================
# FEATURES
# =========================================

X = df[[
    "Close",
    "Volume",
    "RSI",
    "MACD",
    "EMA_20",
    "SMA_20",
    "Returns"
]]



# =========================================
# LABEL
# =========================================

y = df["target"]



# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2
)



# =========================================
# MODEL
# =========================================

model = RandomForestClassifier()



# =========================================
# TRAIN
# =========================================

model.fit(X_train, y_train)



# =========================================
# PREDICT
# =========================================

predictions = model.predict(X_test)



# =========================================
# ACCURACY
# =========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", accuracy)
