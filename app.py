import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# =====================================================
# PAGE
# =====================================================

st.set_page_config(
    page_title="BTC AI Dashboard",
    page_icon="₿",
    layout="wide"
)

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

html, body, [class*="css"]{
    background:#050816;
    color:white;
    font-family:Inter;
}

.block-container{
    padding-top:1rem;
    padding-left:2rem;
    padding-right:2rem;
}

/* TITLE */

.main-title{
    font-size:58px;
    font-weight:800;
    color:white;
    margin-bottom:0px;
}

.sub-title{
    color:#94a3b8;
    font-size:16px;
    margin-bottom:35px;
}

/* CARD */

.card{
    background:#0f172a;

    border-radius:20px;

    padding:28px;

    border:1px solid rgba(255,255,255,0.06);

    box-shadow:
    0 4px 20px rgba(0,0,0,0.35);
}

/* PRICE */

.price-label{
    color:#94a3b8;
    font-size:18px;
}

.price-value{
    color:white;
    font-size:58px;
    font-weight:800;
    margin-top:10px;
}

.green{
    color:#22c55e;
}

.red{
    color:#ef4444;
}

/* SECTION */

.section-title{
    font-size:38px;
    font-weight:700;
    margin-top:35px;
    margin-bottom:20px;
}

/* PREDICTION */

.prediction-up{
    background:#052e16;
    border:1px solid #22c55e;
    border-radius:20px;
    padding:30px;
}

.prediction-down{
    background:#450a0a;
    border:1px solid #ef4444;
    border-radius:20px;
    padding:30px;
}

/* INDICATORS */

.indicator-card{
    background:#0f172a;

    padding:22px;

    border-radius:18px;

    text-align:center;

    border:1px solid rgba(255,255,255,0.06);
}

.indicator-name{
    color:#94a3b8;
    font-size:16px;
}

.indicator-value{
    color:white;
    font-size:42px;
    font-weight:700;
    margin-top:10px;
}

/* SIGNALS */

.signal-neutral{
    background:#082f49;
    border-left:5px solid #3b82f6;
    padding:18px;
    border-radius:14px;
    margin-top:18px;
    font-size:20px;
    font-weight:700;
}

.signal-bull{
    background:#052e16;
    border-left:5px solid #22c55e;
    padding:18px;
    border-radius:14px;
    margin-top:18px;
    font-size:20px;
    font-weight:700;
}

.signal-bear{
    background:#450a0a;
    border-left:5px solid #ef4444;
    padding:18px;
    border-radius:14px;
    margin-top:18px;
    font-size:20px;
    font-weight:700;
}

/* TABLE */

table{
    color:white !important;
}

/* MOBILE */

@media(max-width:768px){

    .main-title{
        font-size:38px;
    }

    .price-value{
        font-size:40px;
    }

    .indicator-value{
        font-size:28px;
    }

}

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.markdown("""
<div class="main-title">
BTC AI Prediction Dashboard
</div>

<div class="sub-title">
Live Bitcoin AI Analysis • Technical Indicators • TradingView Chart
</div>
""", unsafe_allow_html=True)

# =====================================================
# DATA
# =====================================================

btc = yf.download(
    "BTC-USD",
    period="90d",
    interval="1h"
)

btc.columns = btc.columns.get_level_values(0)

df = btc.copy()

# =====================================================
# INDICATORS
# =====================================================

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
    close=df["Close"]
).average_true_range()

df["Returns"] = df["Close"].pct_change()

df["Volatility"] = df["Returns"].rolling(20).std()

# =====================================================
# AI MODEL
# =====================================================

df["Target"] = np.where(
    df["Close"].shift(-1) > df["Close"],
    1,
    0
)

df.dropna(inplace=True)

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

# =====================================================
# LIVE PRICE
# =====================================================

latest_price = float(df["Close"].iloc[-1])

prev_price = float(df["Close"].iloc[-2])

change = latest_price - prev_price

change_percent = (change / prev_price) * 100

# =====================================================
# PRICE UI
# =====================================================

left, right = st.columns([2,1])

with left:

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

with right:

    color = "green" if change >= 0 else "red"

    arrow = "▲" if change >= 0 else "▼"

    st.markdown(f"""
    <div class="card">

    <div class="price-label">
    24H Change
    </div>

    <br>

    <div class="{color}" style="font-size:34px;font-weight:800;">
    {arrow} {change:.2f}
    </div>

    <br>

    <div class="{color}" style="font-size:28px;font-weight:700;">
    {change_percent:.2f}%
    </div>

    </div>
    """, unsafe_allow_html=True)

# =====================================================
# PREDICTION
# =====================================================

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

# =====================================================
# MAIN INDICATORS
# =====================================================

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
    {df["RSI"].iloc[-1]:.2f}
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
    {df["MACD"].iloc[-1]:.2f}
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
    {df["ATR"].iloc[-1]:.2f}
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
    {df["Volatility"].iloc[-1]:.4f}
    </div>

    </div>
    """, unsafe_allow_html=True)

# =====================================================
# SIGNALS
# =====================================================

rsi = df["RSI"].iloc[-1]

if rsi > 70:

    st.markdown(f"""
    <div class="signal-bear">
    RSI Overbought : {rsi:.2f}
    </div>
    """, unsafe_allow_html=True)

elif rsi < 30:

    st.markdown(f"""
    <div class="signal-bull">
    RSI Oversold : {rsi:.2f}
    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown(f"""
    <div class="signal-neutral">
    RSI Neutral : {rsi:.2f}
    </div>
    """, unsafe_allow_html=True)

# MACD SIGNAL

if df["MACD"].iloc[-1] > df["MACD_SIGNAL"].iloc[-1]:

    st.markdown("""
    <div class="signal-bull">
    MACD Bullish Crossover
    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <div class="signal-bear">
    MACD Bearish Crossover
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# TRADINGVIEW
# =====================================================

st.markdown(
    '<div class="section-title">BTC Live Chart</div>',
    unsafe_allow_html=True
)

tradingview = """
<div class="tradingview-widget-container">

<div id="tradingview_chart"></div>

<script type="text/javascript"
src="https://s3.tradingview.com/tv.js"></script>

<script type="text/javascript">

new TradingView.widget({

"autosize": true,

"symbol": "BINANCE:BTCUSDT",

"interval": "60",

"timezone": "Asia/Kolkata",

"theme": "dark",

"style": "1",

"locale": "en",

"toolbar_bg": "#0f172a",

"enable_publishing": false,

"withdateranges": true,

"allow_symbol_change": true,

"details": true,

"hotlist": true,

"calendar": true,

"studies": [
"RSI@tv-basicstudies",
"MACD@tv-basicstudies"
],

"container_id": "tradingview_chart"

});

</script>

</div>
"""

st.components.v1.html(
    tradingview,
    height=900,
    scrolling=False
)

# =====================================================
# EMA CHART
# =====================================================

st.markdown(
    '<div class="section-title">EMA 20 vs EMA 50</div>',
    unsafe_allow_html=True
)

chart_df = df.tail(150)

st.line_chart(
    chart_df[["EMA_20", "EMA_50"]]
)

# =====================================================
# LATEST DATA
# =====================================================

st.markdown(
    '<div class="section-title">Latest BTC Data</div>',
    unsafe_allow_html=True
)

latest_table = df[[
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
    "RSI",
    "MACD",
    "EMA_20",
    "EMA_50",
    "ATR"
]].tail(10)

st.dataframe(
    latest_table,
    use_container_width=True
)
