import requests
import pandas as pd
import ta
import pickle

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)



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



# MACD SIGNAL
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
    "MACD_SIGNAL",
    "EMA_20",
    "EMA_50",
    "SMA_20",
    "Returns",
    "Volatility"
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
    test_size=0.2,
    random_state=42
)



# =========================================
# XGBOOST MODEL
# =========================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)



# =========================================
# TRAIN MODEL
# =========================================

model.fit(
    X_train,
    y_train
)



# =========================================
# PREDICT
# =========================================

predictions = model.predict(
    X_test
)



# =========================================
# ACCURACY
# =========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:")
print(accuracy)



# =========================================
# CLASSIFICATION REPORT
# =========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)



# =========================================
# CONFUSION MATRIX
# =========================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)



# =========================================
# SAVE MODEL
# =========================================

pickle.dump(
    model,
    open("btc_model.pkl", "wb")
)

print("\nModel saved as btc_model.pkl")
