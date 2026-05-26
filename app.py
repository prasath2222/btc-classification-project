import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="BTC AI Dashboard",
    page_icon="₿",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #050816;
    color: white;
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* =========================
TITLE
========================= */

.main-title {
    font-size: 54px;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
}

.sub-title {
    color: #94a3b8;
    font-size: 16px;
    margin-bottom: 30px;
}

/* =========================
CARDS
========================= */

.card {
    background: linear-gradient(
        145deg,
        #0f172a,
        #111827
    );

    border: 1px solid rgba(255,255,255,0.06);

    border-radius: 20px;

    padding: 25px;

    box-shadow:
        0 4px 20px rgba(0,0,0,0.35);
}

/* =========================
PRICE
========================= */

.price-label {
    color: #94a3b8;
    font-size: 18px;
}

.price-value {
    color: white;
    font-size: 58px;
    font-weight: 700;
    margin-top: 12px;
}

.change-up {
    color: #22c55e;
    font-size: 28px;
    font-weight: 700;
}

.change-down {
    color: #ef4444;
    font-size: 28px;
    font-weight: 700;
}

/* =========================
SECTION
========================= */

.section-title {
    font-size: 34px;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 18px;
    color: white;
}

/* =========================
PREDICTION
========================= */

.prediction-up {
    background: linear-gradient(
        145deg,
        #052e16,
        #14532d
    );

    border: 1px solid #22c55e;

    padding: 30px;

    border-radius: 20px;

    margin-top: 20px;
}

.prediction-down {
    background: linear-gradient(
        145deg,
        #3b0712,
        #4c0519
    );

    border: 1px solid #ef4444;

    padding: 30px;

    border-radius: 20px;

    margin-top: 20px;
}

/* =========================
INDICATOR CARD
========================= */

.indicator-card {
    background: #0f172a;

    border-radius: 18px;

    padding: 24px;

    border: 1px solid rgba(255,255,255,0.06);

    text-align: center;
}

.indicator-name {
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 10px;
}

.indicator-value {
    font-size: 40px;
    font-weight: 700;
    color: white;
}

/* =========================
SIGNALS
========================= */

.signal-neutral {
    background: #082f49;
    border-left: 5px solid #3b82f6;
    padding: 18px;
    border-radius: 14px;
    color: white;
    font-size: 20px;
    font-weight: 600;
}

.signal-bullish {
    background: #052e16;
    border-left: 5px solid #22c55e;
    padding: 18px;
    border-radius: 14px;
    color: white;
    font-size: 20px;
    font-weight: 600;
}

.signal-bearish {
    background: #3b0712;
    border-left: 5px solid #ef4444;
    padding: 18px;
    border-radius: 14px;
    color: white;
    font-size: 20px;
    font-weight: 600;
}

/* =========================
TRADINGVIEW
========================= */

.tv-container {
    background: #0f172a;

    padding: 12px;

    border-radius: 20px;

    border: 1px solid rgba(255,255,255,0.06);

    overflow: hidden;

    margin-top: 20px;
}

/* =========================
MOBILE
========================= */

@media (max-width: 768px) {

    .main-title {
        font-size: 34px;
    }

    .section-title {
        font-size: 26px;
    }

    .price-value {
        font-size: 40px;
    }

    .indicator-value {
        font-size: 28px;
    }

}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown("""
<div class="main-title">
BTC AI Prediction Dashboard
</div>

<div class="sub-title">
Live Bitcoin AI Analysis • Technical Indicators • TradingView Chart
</div>
""", unsafe_allow_html=True)

# =========================================================
# DOWNLOAD BTC DATA
# =========================================================

btc = yf.download(
    "BTC-USD",
    period="90d",
    interval="1h"
)

btc.columns = btc.columns.get_level_values(0)

df = btc.copy()

# =========================================================
# INDICATORS
# =========================================================

df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Close"],
    window=14
).rsi()

macd = ta.trend.MACD(df["Close"])

df["MACD"] = macd.macd()
df["MACD_SIGNAL"] = macd.macd_signal()

df["EMA_20"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=20
).ema_indicator()

df["EMA_50"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=50
).ema_indicator()

df["ATR"] = ta.volatility.AverageTrueRange(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14
).average_true_range()

df["Returns"] = df["Close"].pct_change()

df["Volatility"] = df["Returns"].rolling(20).std()

# =========================================================
# AI MODEL
# =========================================================

df["Target"] = np.where(
    df["Close"].shift(-1) > df["Close"],
    1,
    0
)

df = df.dropna()

features = [
    "RSI",
    "MACD",
    "MACD_SIGNAL",
    "EMA_20",
    "EMA_50",
    "ATR",
    "Returns",
    "Volatility"
]

X = df[features]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

model = XGBClassifier()

model.fit(X_train, y_train)

latest = df[features].iloc[-1:]

prediction = model.predict(latest)[0]

confidence = model.predict_proba(latest)[0].max() * 100

# =========================================================
# LIVE PRICE
# =========================================================

latest_price = float(df["Close"].iloc[-1])

prev_price = float(df["Close"].iloc[-2])

change = latest_price - prev_price

change_percent = (change / prev_price) * 100

# =========================================================
# PRICE UI
# =========================================================

col1, col2 = st.columns([2, 1])

with col1:

    st.markdown(f"""
    <div class="card">

        <div class="price-label">
            BTC/USD
        </div>

        <div class="price-value">
            ${latest_price:,.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

with col2:

    change_class = (
        "change-up"
        if change >= 0
        else "change-down"
    )

    arrow = "▲" if change >= 0 else "▼"

    st.markdown(f"""
    <div class="card">

        <div class="price-label">
            24H Change
        </div>

        <br>

        <div class="{change_class}">
            {arrow} {change:.2f}
        </div>

        <br>

        <div class="{change_class}">
            {change_percent:.2f}%
        </div>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# AI PREDICTION
# =========================================================

if prediction == 1:

    pred_class = "prediction-up"
    pred_text = "BTC may go UP"

else:

    pred_class = "prediction-down"
    pred_text = "BTC may go DOWN"

st.markdown(f"""
<div class="{pred_class}">

<h2>AI Prediction</h2>

<h1>{pred_text}</h1>

<h3>
Confidence : {confidence:.2f}%
</h3>

</div>
""", unsafe_allow_html=True)

# =========================================================
# MAIN INDICATORS
# =========================================================

st.markdown(
    '<div class="section-title">Main Indicators</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown(f"""
    <div class="indicator-card">

        <div class="indicator-name">
            RSI
        </div>

        <div class="indicator-value">
            {df['RSI'].iloc[-1]:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

with c2:

    st.markdown(f"""
    <div class="indicator-card">

        <div class="indicator-name">
            MACD
        </div>

        <div class="indicator-value">
            {df['MACD'].iloc[-1]:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

with c3:

    st.markdown(f"""
    <div class="indicator-card">

        <div class="indicator-name">
            ATR
        </div>

        <div class="indicator-value">
            {df['ATR'].iloc[-1]:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

with c4:

    st.markdown(f"""
    <div class="indicator-card">

        <div class="indicator-name">
            Volatility
        </div>

        <div class="indicator-value">
            {df['Volatility'].iloc[-1]:.4f}
        </div>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# RSI SIGNAL
# =========================================================

rsi = df["RSI"].iloc[-1]

if rsi > 70:

    signal_class = "signal-bearish"
    signal_text = f"RSI Overbought : {rsi:.2f}"

elif rsi < 30:

    signal_class = "signal-bullish"
    signal_text = f"RSI Oversold : {rsi:.2f}"

else:

    signal_class = "signal-neutral"
    signal_text = f"RSI Neutral : {rsi:.2f}"

st.markdown(f"""
<div class="{signal_class}">
{signal_text}
</div>
""", unsafe_allow_html=True)

# =========================================================
# MACD SIGNAL
# =========================================================

if df["MACD"].iloc[-1] > df["MACD_SIGNAL"].iloc[-1]:

    macd_class = "signal-bullish"
    macd_text = "MACD Bullish Crossover"

else:

    macd_class = "signal-bearish"
    macd_text = "MACD Bearish Crossover"

st.markdown(f"""
<div class="{macd_class}">
{macd_text}
</div>
""", unsafe_allow_html=True)

# =========================================================
# TRADINGVIEW CHART
# =========================================================

st.markdown(
    '<div class="section-title">BTC Live Chart</div>',
    unsafe_allow_html=True
)

tradingview_chart = """

<div class="tv-container">

<div class="tradingview-widget-container">

<div id="tradingview_chart"></div>

<script type="text/javascript"
src="https://s3.tradingview.com/tv.js"></script>

<script type="text/javascript">

new TradingView.widget({

    "width": "100%",

    "height": 850,

    "symbol": "BINANCE:BTCUSDT",

    "interval": "60",

    "timezone": "Asia/Kolkata",

    "theme": "dark",

    "style": "1",

    "locale": "en",

    "toolbar_bg": "#0f172a",

    "enable_publishing": false,

    "allow_symbol_change": true,

    "withdateranges": true,

    "hide_side_toolbar": false,

    "save_image": true,

    "details": true,

    "calendar": true,

    "container_id": "tradingview_chart"

});

</script>

</div>

</div>

"""

st.components.v1.html(
    tradingview_chart,
    height=900,
    scrolling=False
)
