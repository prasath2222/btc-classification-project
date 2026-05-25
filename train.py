import pandas as pd
import ta
import pickle
import yfinance as yf

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



print("Downloading BTC data...")

df = yf.download(
    "BTC-USD",
    period="700d",
    interval="1h",
    auto_adjust=True
)



# ====================================
# FIX MULTI INDEX
# ====================================

if isinstance(df.columns, pd.MultiIndex):

    df.columns = df.columns.get_level_values(0)



df.reset_index(inplace=True)



# ====================================
# KEEP COLUMNS
# ====================================

df = df[[
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]].copy()



# ====================================
# FLOAT
# ====================================

for col in df.columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )



df.dropna(inplace=True)



# ====================================
# SERIES
# ====================================

close = df["Close"].squeeze()

high = df["High"].squeeze()

low = df["Low"].squeeze()

volume = df["Volume"].squeeze()



print("Creating indicators...")



# ====================================
# RSI
# ====================================

df["RSI"] = ta.momentum.RSIIndicator(
    close=close,
    window=14
).rsi()



# ====================================
# MACD
# ====================================

macd = ta.trend.MACD(close=close)

df["MACD"] = macd.macd()

df["MACD_SIGNAL"] = macd.macd_signal()

df["MACD_DIFF"] = (
    df["MACD"] -
    df["MACD_SIGNAL"]
)



# ====================================
# EMA
# ====================================

df["EMA_20"] = ta.trend.EMAIndicator(
    close=close,
    window=20
).ema_indicator()



df["EMA_50"] = ta.trend.EMAIndicator(
    close=close,
    window=50
).ema_indicator()



# ====================================
# SMA
# ====================================

df["SMA_20"] = ta.trend.SMAIndicator(
    close=close,
    window=20
).sma_indicator()



# ====================================
# RETURNS
# ====================================

df["Returns"] = (
    close.pct_change()
)



# ====================================
# VOLATILITY
# ====================================

df["Volatility"] = (
    high - low
) / close



# ====================================
# ATR
# ====================================

df["ATR"] = ta.volatility.AverageTrueRange(
    high=high,
    low=low,
    close=close,
    window=14
).average_true_range()



# ====================================
# BOLLINGER
# ====================================

bb = ta.volatility.BollingerBands(
    close=close,
    window=20
)

df["BB_HIGH"] = bb.bollinger_hband()

df["BB_LOW"] = bb.bollinger_lband()

df["BB_WIDTH"] = (
    df["BB_HIGH"] -
    df["BB_LOW"]
)



# ====================================
# STOCH
# ====================================

stoch = ta.momentum.StochasticOscillator(
    high=high,
    low=low,
    close=close,
    window=14
)

df["STOCH"] = stoch.stoch()

df["STOCH_SIGNAL"] = (
    stoch.stoch_signal()
)



# ====================================
# VOLUME CHANGE
# ====================================

df["Volume_Change"] = (
    volume.pct_change()
)



# ====================================
# TARGET
# ====================================

df["Target"] = (
    df["Close"].shift(-1) >
    df["Close"]
).astype(int)



# ====================================
# CLEAN
# ====================================

df.dropna(inplace=True)



print("Preparing features...")



# ====================================
# FEATURES
# ====================================

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
    "Volume_Change"
]



X = df[features]

y = df["Target"]



# ====================================
# SPLIT
# ====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)



print("Training AI model...")



# ====================================
# MODEL
# ====================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)



model.fit(
    X_train,
    y_train
)



# ====================================
# TEST
# ====================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)



print("\nAccuracy:", accuracy)



# ====================================
# SAVE
# ====================================

pickle.dump(
    model,
    open("btc_model.pkl", "wb")
)



print("\nModel saved successfully!")
