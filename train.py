# =========================================================
# BTC AI TRAINING MODEL
# =========================================================

import pandas as pd
import numpy as np
import yfinance as yf
import ta
import pickle

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# =========================================================
# DOWNLOAD BTC DATA
# =========================================================

print("Downloading BTC data...")

df = yf.download(
    "BTC-USD",
    period="730d",
    interval="1h"
)

# =========================================================
# FIX MULTI INDEX
# =========================================================

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# =========================================================
# INDICATORS
# =========================================================

print("Creating indicators...")

# RSI
df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Close"],
    window=14
).rsi()

# MACD
macd = ta.trend.MACD(
    close=df["Close"]
)

df["MACD"] = macd.macd()

df["MACD_SIGNAL"] = (
    macd.macd_signal()
)

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

# ATR
df["ATR"] = ta.volatility.AverageTrueRange(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14
).average_true_range()

# RETURNS
df["Returns"] = (
    df["Close"].pct_change()
)

# VOLATILITY
df["Volatility"] = (
    df["Returns"].rolling(20).std()
)

# TREND
df["Trend"] = (
    df["EMA_20"] > df["EMA_50"]
).astype(int)

# =========================================================
# TARGET
# =========================================================

print("Preparing target...")

# Predict next 3 candles
df["Future_Close"] = (
    df["Close"].shift(-3)
)

df["Target"] = (
    df["Future_Close"] > df["Close"]
).astype(int)

# =========================================================
# CLEAN DATA
# =========================================================

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

df.dropna(inplace=True)

# =========================================================
# FEATURES
# =========================================================

features = [

    "RSI",

    "MACD",
    "MACD_SIGNAL",

    "EMA_20",
    "EMA_50",

    "ATR",

    "Returns",

    "Volatility",

    "Trend"
]

# =========================================================
# FORCE NUMERIC
# =========================================================

for col in features:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df.dropna(inplace=True)

# =========================================================
# X AND Y
# =========================================================

X = df[features]

y = df["Target"]

# =========================================================
# TIME SERIES SPLIT
# =========================================================

split = int(len(df) * 0.8)

X_train = X.iloc[:split]

X_test = X.iloc[split:]

y_train = y.iloc[:split]

y_test = y.iloc[split:]

# =========================================================
# MODEL
# =========================================================

print("Training AI model...")

model = XGBClassifier(

    n_estimators=500,

    max_depth=4,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    gamma=0.3,

    min_child_weight=5,

    reg_alpha=0.5,

    reg_lambda=2,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42
)

# =========================================================
# TRAIN
# =========================================================

model.fit(
    X_train,
    y_train
)

# =========================================================
# PREDICT
# =========================================================

predictions = model.predict(
    X_test
)

# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n========================")
print("MODEL ACCURACY:", accuracy)
print("========================")

# =========================================================
# SAVE MODEL
# =========================================================

pickle.dump(
    model,
    open("btc_model.pkl", "wb")
)

print("\nModel saved successfully!")
