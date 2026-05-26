import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pickle
from streamlit_autorefresh import st_autorefresh

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="BTC AI Dashboard",
    layout="wide"
)

# ============================================
# AUTO REFRESH
# ============================================

st_autorefresh(
    interval=10000,
    key="btc_refresh"
)

# ============================================
# LOAD MODEL
# ============================================

model = pickle.load(
    open("btc_model.pkl", "rb")
)

# ============================================
# TITLE
# ============================================

st.title("BTC AI Prediction Dashboard")

# ============================================
# DOWNLOAD BTC DATA
# ============================================

df = yf.download(
    "BTC-USD",
    period="60d",
    interval="1h"
)

# ============================================
# FIX MULTIINDEX
# ============================================

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# ============================================
# RESET INDEX
# ============================================

df.reset_index(inplace=True)

# ============================================
# TECHNICAL INDICATORS
# ============================================

# RSI
df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Close"],
    window=14
).rsi()

# MACD
macd = ta.trend.MACD(df["Close"])

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

# Bollinger Bands
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
df["Volatility"] = df["Returns"].rolling(
    24
).std()

# VOLUME CHANGE
df["Volume_Change"] = df["Volume"].pct_change()

# TREND
df["Trend"] = np.where(
    df["EMA_20"] > df["EMA_50"],
    1,
    0
)

# ============================================
# REMOVE NAN
# ============================================

df.dropna(inplace=True)

# ============================================
# FEATURE COLUMNS
# ============================================

feature_columns = [
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

# ============================================
# LATEST DATA
# ============================================

latest = df[feature_columns].tail(1)

# ============================================
# PREDICTION
# ============================================

prediction = model.predict(latest)[0]

probability = model.predict_proba(latest)[0]

confidence = round(
    max(probability) * 100,
    2
)

# ============================================
# LIVE PRICE
# ============================================

live_price = round(
    float(df["Close"].iloc[-1]),
    2
)

previous_price = round(
    float(df["Close"].iloc[-2]),
    2
)

price_change = round(
    live_price - previous_price,
    2
)

# ============================================
# PRICE DISPLAY
# ============================================

st.subheader("Live BTC Price")

st.metric(
    label="BTC/USD",
    value=f"${live_price}",
    delta=price_change
)

# ============================================
# AI PREDICTION
# ============================================

if prediction == 1:

    st.success(
        f"AI Prediction: BTC may go UP | Confidence: {confidence}%"
    )

else:

    st.error(
        f"AI Prediction: BTC may go DOWN | Confidence: {confidence}%"
    )

# ============================================
# RSI STATUS
# ============================================

latest_rsi = df["RSI"].iloc[-1]

if latest_rsi > 70:

    st.warning(
        f"RSI Overbought : {latest_rsi:.2f}"
    )

elif latest_rsi < 30:

    st.success(
        f"RSI Oversold : {latest_rsi:.2f}"
    )

else:

    st.info(
        f"RSI Neutral : {latest_rsi:.2f}"
    )

# ============================================
# MACD STATUS
# ============================================

latest_macd = df["MACD"].iloc[-1]

latest_signal = df["MACD_SIGNAL"].iloc[-1]

if latest_macd > latest_signal:

    st.success(
        "MACD Bullish Crossover"
    )

else:

    st.error(
        "MACD Bearish Crossover"
    )

# ============================================
# TRADINGVIEW LIVE CHART
# ============================================

st.markdown("## BTC Live Candlestick Chart")

tradingview_widget = """
<!-- TradingView Widget BEGIN -->
<div class="tradingview-widget-container">
  <div id="tradingview_btc"></div>

  <script type="text/javascript"
  src="https://s3.tradingview.com/tv.js"></script>

  <script type="text/javascript">

  new TradingView.widget(
  {
    "width": "100%",
    "height": 700,
    "symbol": "BINANCE:BTCUSDT",
    "interval": "60",
    "timezone": "Asia/Kolkata",
    "theme": "dark",
    "style": "1",
    "locale": "en",
    "toolbar_bg": "#0b1220",
    "enable_publishing": false,
    "allow_symbol_change": true,
    "withdateranges": true,
    "hide_side_toolbar": false,
    "details": true,
    "hotlist": true,
    "calendar": true,
    "container_id": "tradingview_btc"
  }
  );

  </script>
</div>
<!-- TradingView Widget END -->
"""

st.components.v1.html(
    tradingview_widget,
    height=720
)

# ============================================
# LATEST DATA TABLE
# ============================================

st.markdown("## Latest BTC Data")

st.dataframe(
    df.tail(10)
)
