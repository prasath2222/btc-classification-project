# =========================================================
# BEST ADVANCED TRAIN.PY
# MARKET STRUCTURE + MOMENTUM + BTC CORRELATION
# =========================================================

import ccxt
import pandas as pd
import numpy as np
import ta
import joblib

from xgboost import XGBClassifier
from xgboost import XGBRegressor

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
    limit=15000
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
    limit=15000
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
# FUTURE TARGET
# =========================================================

future_period = 6

df["Future_Close"] = (
    df["close"]
    .shift(-future_period)
)

future_return = (
    (df["Future_Close"] - df["close"])
    / df["close"]
)

# =========================================================
# MULTI CLASS TARGET
# =========================================================

conditions = [

    future_return > 0.04,

    (
        (future_return > 0.01)
        &
        (future_return <= 0.04)
    ),

    (
        (future_return >= -0.01)
        &
        (future_return <= 0.01)
    ),

    (
        (future_return < -0.01)
        &
        (future_return >= -0.04)
    ),

    future_return < -0.04

]

choices = [
    2,
    1,
    0,
    -1,
    -2
]

df["Target"] = np.select(
    conditions,
    choices,
    default=0
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

features = [

    "close",
    "volume",

    "BTC_Close",
    "BTC_Return",

    "RSI",

    "MACD",
    "MACD_SIGNAL",
    "MACD_DIFF",

    "EMA20",
    "EMA50",
    "EMA200",

    "ATR",

    "BB_HIGH",
    "BB_LOW",
    "BB_WIDTH",

    "Returns",

    "Volatility",

    "Momentum5",
    "Momentum10",

    "Higher_High",
    "Higher_Low",

    "Breakout",

    "Volume_Spike",

    "Body",
    "Upper_Wick",
    "Lower_Wick",

    "Bull_Trend"
]

X = df[features]

# =========================================================
# TARGETS
# =========================================================

y_class = df["Target"]

y_reg = df["Future_Close"]

# =========================================================
# CLASSIFIER
# =========================================================

classifier = XGBClassifier(

    n_estimators=400,

    max_depth=10,

    learning_rate=0.03,

    subsample=0.9,

    colsample_bytree=0.9,

    gamma=0.1,

    random_state=42,

    eval_metric="mlogloss"
)

# =========================================================
# REGRESSOR
# =========================================================

regressor = XGBRegressor(

    n_estimators=400,

    max_depth=10,

    learning_rate=0.03,

    subsample=0.9,

    colsample_bytree=0.9,

    random_state=42
)

# =========================================================
# TRAIN
# =========================================================

classifier.fit(
    X,
    y_class
)

regressor.fit(
    X,
    y_reg
)

# =========================================================
# SAVE
# =========================================================

joblib.dump(
    classifier,
    "rf_classifier.pkl"
)

joblib.dump(
    regressor,
    "rf_regressor.pkl"
)

joblib.dump(
    features,
    "features.pkl"
)

print("\n=================================")
print("ADVANCED AI MODELS TRAINED")
print("=================================")

print("\nFILES SAVED:")
print("rf_classifier.pkl")
print("rf_regressor.pkl")
print("features.pkl")
print("=================================")
