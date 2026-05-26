# =========================================================
# BEST ADVANCED PREDICT.PY
# LIVE MARKET PREDICTION
# =========================================================

import ccxt
import pandas as pd
import numpy as np
import ta
import joblib

# =========================================================
# LOAD MODELS
# =========================================================

classifier = joblib.load(
    "rf_classifier.pkl"
)

regressor = joblib.load(
    "rf_regressor.pkl"
)

features = joblib.load(
    "features.pkl"
)

# =========================================================
# EXCHANGE
# =========================================================

exchange = ccxt.binance()

# =========================================================
# RNDR DATA
# =========================================================

bars = exchange.fetch_ohlcv(
    'RENDER/USDT',
    timeframe='1h',
    limit=300
)

df = pd.DataFrame(
    bars,
    columns=[
        'timestamp',
        'open',
        'high',
        'low',
        'close',
        'volume'
    ]
)

# =========================================================
# BTC DATA
# =========================================================

btc_bars = exchange.fetch_ohlcv(
    'BTC/USDT',
    timeframe='1h',
    limit=300
)

btc_df = pd.DataFrame(
    btc_bars,
    columns=[
        'timestamp',
        'btc_open',
        'btc_high',
        'btc_low',
        'btc_close',
        'btc_volume'
    ]
)

# =========================================================
# BTC FEATURES
# =========================================================

df["BTC_Close"] = btc_df["btc_close"]

df["BTC_Return"] = (
    df["BTC_Close"]
    .pct_change()
)

# =========================================================
# RSI
# =========================================================

df["RSI"] = ta.momentum.RSIIndicator(
    close=df["close"],
    window=14
).rsi()

# =========================================================
# MACD
# =========================================================

macd = ta.trend.MACD(
    close=df["close"]
)

df["MACD"] = macd.macd()

df["MACD_SIGNAL"] = (
    macd.macd_signal()
)

df["MACD_DIFF"] = (
    macd.macd_diff()
)

# =========================================================
# EMA
# =========================================================

df["EMA20"] = ta.trend.EMAIndicator(
    close=df["close"],
    window=20
).ema_indicator()

df["EMA50"] = ta.trend.EMAIndicator(
    close=df["close"],
    window=50
).ema_indicator()

df["EMA200"] = ta.trend.EMAIndicator(
    close=df["close"],
    window=200
).ema_indicator()

# =========================================================
# ATR
# =========================================================

df["ATR"] = (
    ta.volatility.AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    )
    .average_true_range()
)

# =========================================================
# BOLLINGER
# =========================================================

bb = ta.volatility.BollingerBands(
    close=df["close"],
    window=20
)

df["BB_HIGH"] = (
    bb.bollinger_hband()
)

df["BB_LOW"] = (
    bb.bollinger_lband()
)

df["BB_WIDTH"] = (
    bb.bollinger_wband()
)

# =========================================================
# RETURNS
# =========================================================

df["Returns"] = (
    df["close"]
    .pct_change()
)

# =========================================================
# VOLATILITY
# =========================================================

df["Volatility"] = (
    df["Returns"]
    .rolling(24)
    .std()
)

# =========================================================
# MOMENTUM
# =========================================================

df["Momentum5"] = (
    df["close"]
    - df["close"].shift(5)
)

df["Momentum10"] = (
    df["close"]
    - df["close"].shift(10)
)

# =========================================================
# MARKET STRUCTURE
# =========================================================

df["Higher_High"] = (
    df["high"]
    > df["high"].shift(1)
).astype(int)

df["Higher_Low"] = (
    df["low"]
    > df["low"].shift(1)
).astype(int)

# =========================================================
# BREAKOUT
# =========================================================

df["Breakout"] = (
    df["close"]
    >
    df["high"]
    .rolling(20)
    .max()
    .shift(1)
).astype(int)

# =========================================================
# VOLUME SPIKE
# =========================================================

df["Volume_MA"] = (
    df["volume"]
    .rolling(20)
    .mean()
)

df["Volume_Spike"] = (
    df["volume"]
    >
    (df["Volume_MA"] * 2)
).astype(int)

# =========================================================
# CANDLE FEATURES
# =========================================================

df["Body"] = abs(
    df["close"] - df["open"]
)

df["Upper_Wick"] = (
    df["high"]
    -
    df[["close", "open"]]
    .max(axis=1)
)

df["Lower_Wick"] = (
    df[["close", "open"]]
    .min(axis=1)
    -
    df["low"]
)

# =========================================================
# TREND REGIME
# =========================================================

df["Bull_Trend"] = np.where(
    (
        (df["EMA20"] > df["EMA50"])
        &
        (df["EMA50"] > df["EMA200"])
    ),
    1,
    0
)

# =========================================================
# CLEAN
# =========================================================

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

df.dropna(inplace=True)

# =========================================================
# LATEST DATA
# =========================================================

latest = df.iloc[-1]

X_live = pd.DataFrame(
    [[latest[f] for f in features]],
    columns=features
)

# =========================================================
# PREDICTIONS
# =========================================================

class_prediction = classifier.predict(
    X_live
)[0]

probabilities = classifier.predict_proba(
    X_live
)[0]

confidence = round(
    np.max(probabilities) * 100,
    2
)

future_price = regressor.predict(
    X_live
)[0]

current_price = latest["close"]

# =========================================================
# LABELS
# =========================================================

labels = {

    4: "STRONG BUY 🚀",
    3: "BUY 📈",
    2: "SIDEWAYS ➖",
    1: "SELL 📉",
    0: "STRONG SELL 🔥"

}

signal = labels[class_prediction]

# =========================================================
# PRINT RESULTS
# =========================================================

print("\n===================================")
print("LIVE AI MARKET PREDICTION")
print("===================================")

print(f"\nCurrent Price : ${current_price:.4f}")

print(f"\nPredicted Price : ${future_price:.4f}")

print(f"\nAI Signal : {signal}")

print(f"\nConfidence : {confidence}%")

print("\n===================================")
