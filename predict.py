import ccxt
import pandas as pd
import ta
import joblib
import numpy as np

# ==========================================
# LOAD MODELS
# ==========================================

classifier = joblib.load("rf_classifier.pkl")
regressor = joblib.load("rf_regressor.pkl")

# ==========================================
# GET LIVE DATA
# ==========================================

exchange = ccxt.binance()

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

# ==========================================
# INDICATORS
# ==========================================

df["RSI"] = ta.momentum.RSIIndicator(
    df["close"],
    window=14
).rsi()

macd = ta.trend.MACD(df["close"])

df["MACD"] = macd.macd()
df["MACD_SIGNAL"] = macd.macd_signal()

df["EMA20"] = ta.trend.EMAIndicator(
    df["close"],
    window=20
).ema_indicator()

df["EMA50"] = ta.trend.EMAIndicator(
    df["close"],
    window=50
).ema_indicator()

df["EMA200"] = ta.trend.EMAIndicator(
    df["close"],
    window=200
).ema_indicator()

df["ATR"] = ta.volatility.AverageTrueRange(
    df["high"],
    df["low"],
    df["close"]
).average_true_range()

df["Returns"] = df["close"].pct_change()

df["Volatility"] = (
    df["Returns"]
    .rolling(24)
    .std()
)

df["Momentum"] = (
    df["close"] -
    df["close"].shift(5)
)

df.dropna(inplace=True)

# ==========================================
# FEATURES
# ==========================================

latest = pd.DataFrame([{
    "close": df["close"].iloc[-1],
    "volume": df["volume"].iloc[-1],
    "RSI": df["RSI"].iloc[-1],
    "MACD": df["MACD"].iloc[-1],
    "MACD_SIGNAL": df["MACD_SIGNAL"].iloc[-1],
    "EMA20": df["EMA20"].iloc[-1],
    "EMA50": df["EMA50"].iloc[-1],
    "EMA200": df["EMA200"].iloc[-1],
    "ATR": df["ATR"].iloc[-1],
    "Volatility": df["Volatility"].iloc[-1],
    "Momentum": df["Momentum"].iloc[-1]
}])

# ==========================================
# PREDICTION
# ==========================================

prediction = classifier.predict(latest)[0]

predicted_price = regressor.predict(latest)[0]

prob = classifier.predict_proba(latest)[0]

confidence = round(
    np.max(prob) * 100,
    2
)

# ==========================================
# SIGNAL LOGIC
# ==========================================

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

# ==========================================
# EXTRA TREND FILTER
# ==========================================

price = df["close"].iloc[-1]

ema20 = df["EMA20"].iloc[-1]
ema50 = df["EMA50"].iloc[-1]
ema200 = df["EMA200"].iloc[-1]

rsi = df["RSI"].iloc[-1]

macd_now = df["MACD"].iloc[-1]

if (
    price > ema20
    and ema20 > ema50
    and ema50 > ema200
    and rsi > 55
    and macd_now > 0
):
    signal = "STRONG BUY"

elif (
    price < ema20
    and ema20 < ema50
    and ema50 < ema200
    and rsi < 45
    and macd_now < 0
):
    signal = "STRONG SELL"

# ==========================================
# PRINT RESULT
# ==========================================

print("\n============================")
print("LIVE AI PREDICTION")
print("============================")

print(f"\nCurrent Price : ${price:.4f}")

print(f"\nPredicted Price : ${predicted_price:.4f}")

print(f"\nSignal : {signal}")

print(f"\nConfidence : {confidence}%")

print("\n============================")
