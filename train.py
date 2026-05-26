!pip install ta # Install the 'ta' library

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pickle

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ======================================================
# DOWNLOAD BTC DATA
# ======================================================

df = yf.download(
    "BTC-USD",
    period="730d",
    interval="1h"
)

# FIX MULTI INDEX
df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

# ======================================================
# TECHNICAL INDICATORS
# ======================================================

# RSI
df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Close"],
    window=14
).rsi()

# MACD
macd = ta.trend.MACD(close=df["Close"])

df["MACD"] = macd.macd()
df["MACD_SIGNAL"] = macd.macd_signal()
df["MACD_DIFF"] = macd.macd_diff()

# EMA
df["EMA_20"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=20
).ema_indicator()

df["EMA_50"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=50
).ema_indicator()

# SMA
df["SMA_20"] = ta.trend.SMAIndicator(
    close=df["Close"],
    window=20
).sma_indicator()

# BOLLINGER BANDS
bb = ta.volatility.BollingerBands(
    close=df["Close"],
    window=20,
    window_dev=2
)

df["BB_HIGH"] = bb.bollinger_hband()
df["BB_LOW"] = bb.bollinger_lband()
df["BB_WIDTH"] = bb.bollinger_wband()

# ATR
df["ATR"] = ta.volatility.AverageTrueRange(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14
).average_true_range()

# STOCHASTIC
stoch = ta.momentum.StochasticOscillator(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14,
    smooth_window=3
)

df["STOCH"] = stoch.stoch()
df["STOCH_SIGNAL"] = stoch.stoch_signal()

# RETURNS
df["Returns"] = df["Close"].pct_change()

# VOLATILITY
df["Volatility"] = df["Returns"].rolling(24).std()

# VOLUME CHANGE
df["Volume_Change"] = df["Volume"].pct_change()

# TREND
df["Trend"] = np.where(
    df["EMA_20"] > df["EMA_50"],
    1,
    0
)

# ======================================================
# TARGET
# ======================================================

df["Future_Return"] = (
    df["Close"].shift(-6) / df["Close"] - 1
)

df["Target"] = (
    df["Future_Return"] > 0.01
).astype(int)

# ======================================================
# CLEAN DATA
# ======================================================

# Replace inf values with NaN before dropping rows
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

# ======================================================
# FEATURES
# ======================================================

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
    "Volume_Change",
    "Trend"
]

X = df[features]
y = df["Target"]

# ======================================================
# TRAIN TEST SPLIT
# ======================================================

split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# ======================================================
# MODEL
# ======================================================

model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    random_state=42,
    eval_metric="logloss"
)

# ======================================================
# TRAIN
# ======================================================

model.fit(
    X_train,
    y_train
)

# ======================================================
# PREDICT
# ======================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n==========================")
print("MODEL ACCURACY:", accuracy)
print("==========================")

# ======================================================
# SAVE MODEL
# ======================================================

pickle.dump(
    model,
    open("btc_model.pkl", "wb")
)

print("\nModel saved successfully!")
