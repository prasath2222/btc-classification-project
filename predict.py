# =========================================================
# ADVANCED LIVE AI PREDICTION ENGINE
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
    limit=500
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
    limit=500
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
# FEATURES
# =========================================================

latest = df[features].iloc[-1:]

# =========================================================
# PREDICTIONS
# =========================================================

prediction = classifier.predict(
    latest
)[0]

predicted_price = regressor.predict(
    latest
)[0]

probabilities = classifier.predict_proba(
    latest
)[0]

confidence = round(
    np.max(probabilities) * 100,
    2
)

# =========================================================
# SIGNAL MAP
# =========================================================

signal_map = {

    2: "STRONG BUY",

    1: "BUY",

    0: "SIDEWAYS",

    -1: "SELL",

    -2: "STRONG SELL"
}

signal = signal_map.get(
    prediction,
    "SIDEWAYS"
)

# =========================================================
# EXTRA MOMENTUM FILTER
# =========================================================

price = df["close"].iloc[-1]

ema20 = df["EMA20"].iloc[-1]
ema50 = df["EMA50"].iloc[-1]
ema200 = df["EMA200"].iloc[-1]

rsi = df["RSI"].iloc[-1]

macd_now = df["MACD"].iloc[-1]

breakout = df["Breakout"].iloc[-1]

volume_spike = df["Volume_Spike"].iloc[-1]

# =========================================================
# BULLISH OVERRIDE
# =========================================================

if (

    price > ema20
    and ema20 > ema50
    and ema50 > ema200

    and rsi > 55

    and macd_now > 0

    and breakout == 1

):

    signal = "STRONG BUY"

# =========================================================
# BEARISH OVERRIDE
# =========================================================

elif (

    price < ema20
    and ema20 < ema50
    and ema50 < ema200

    and rsi < 45

    and macd_now < 0

):

    signal = "STRONG SELL"

# =========================================================
# VOLUME BOOST
# =========================================================

if volume_spike == 1 and "BUY" in signal:
    confidence += 5

if confidence > 99:
    confidence = 99

# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n===================================")
print("LIVE RNDR AI PREDICTION")
print("===================================")

print(f"\nCurrent Price : ${price:.4f}")

print(f"\nPredicted Price : ${predicted_price:.4f}")

print(f"\nSignal : {signal}")

print(f"\nConfidence : {confidence}%")

print("===================================")
