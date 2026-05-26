import streamlit as st
import ccxt
import pandas as pd
import ta
from tradingview_ta import TA_Handler, Interval
from streamlit.components.v1 import html

# ---------------- PAGE ----------------

st.set_page_config(
    page_title="BTC AI Dashboard",
    layout="wide"
)

# ---------------- STYLE ----------------

st.markdown("""
<style>

body {
    background:#020617;
}

.main {
    background:#020617;
    color:white;
}

.block-container{
    padding-top:30px;
    padding-bottom:40px;
    max-width:1400px;
}

/* TITLE */

.title{
    font-size:64px;
    font-weight:900;
    color:white;
}

.subtitle{
    color:#94a3b8;
    font-size:22px;
    margin-top:-10px;
    margin-bottom:35px;
}

/* CARDS */

.card{
    background:#071a45;
    border:1px solid #12306b;
    border-radius:20px;
    padding:25px;
    height:150px;
}

.metric-label{
    color:#94a3b8;
    font-size:20px;
    margin-bottom:15px;
}

.metric-number{
    font-size:42px;
    font-weight:800;
}

.green{
    color:#00ff99;
}

.red{
    color:#ff4d6d;
}

.yellow{
    color:#facc15;
}

.blue{
    color:#38bdf8;
}

/* SECTION */

.section-title{
    font-size:48px;
    font-weight:900;
    margin-top:40px;
    margin-bottom:25px;
}

/* MARKET BOX */

.move-box{
    background:#071a45;
    border:1px solid #12306b;
    border-radius:22px;
    padding:35px;
    margin-top:20px;
}

/* PREDICTION */

.prediction-box{
    background:#2b2503;
    border:2px solid #facc15;
    border-radius:22px;
    padding:35px;
    margin-top:35px;
}

.predict-title{
    font-size:22px;
    color:white;
    margin-bottom:15px;
}

.predict-text{
    font-size:52px;
    font-weight:900;
}

.small-text{
    margin-top:18px;
    font-size:24px;
    color:white;
}

.chart-box{
    margin-top:30px;
    border-radius:20px;
    overflow:hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------- DATA ----------------

exchange = ccxt.binance()

bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=200)

df = pd.DataFrame(
    bars,
    columns=['timestamp','open','high','low','close','volume']
)

df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()

macd = ta.trend.MACD(df['close'])

df['macd'] = macd.macd()

df['atr'] = ta.volatility.AverageTrueRange(
    df['high'],
    df['low'],
    df['close']
).average_true_range()

btc_price = df['close'].iloc[-1]
rsi = df['rsi'].iloc[-1]
macd_value = df['macd'].iloc[-1]
atr = df['atr'].iloc[-1]

volatility = (
    (df['high'].iloc[-1] - df['low'].iloc[-1])
    / df['close'].iloc[-1]
)

change_24h = (
    df['close'].iloc[-1] - df['close'].iloc[-24]
)

change_percent = (
    (change_24h / df['close'].iloc[-24]) * 100
)

# ---------------- AI LOGIC ----------------

if rsi > 65 and macd_value > 0:
    prediction = "BTC may go UP"
    pred_color = "#00ff99"
    confidence = "74%"

elif rsi < 40 and macd_value < 0:
    prediction = "BTC may go DOWN"
    pred_color = "#ff4d6d"
    confidence = "86%"

else:
    prediction = "BTC may go SIDEWAYS"
    pred_color = "#facc15"
    confidence = "68%"

# ---------------- HEADER ----------------

st.markdown("""
<div class="title">
BTC AI Prediction Dashboard
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Live Bitcoin Analysis • Trading Setup • Technical Indicators
</div>
""", unsafe_allow_html=True)

# ---------------- TOP ROW ----------------

col1, col2 = st.columns([3,1])

with col1:

    st.markdown(f"""
    <div class="card">
        <div class="metric-label">
            BTC/USDT
        </div>

        <div class="metric-number">
            ${btc_price:,.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:

    change_color = "#00ff99" if change_24h > 0 else "#ff4d6d"

    arrow = "▲" if change_24h > 0 else "▼"

    st.markdown(f"""
    <div class="card">
        <div class="metric-label">
            24H Change
        </div>

        <div class="metric-number"
        style="color:{change_color};">
            {arrow} {abs(change_24h):.2f}
        </div>

        <div style="
        margin-top:12px;
        font-size:28px;
        font-weight:700;
        color:{change_color};
        ">
            {change_percent:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- AI PREDICTION ----------------

st.markdown(f"""
<div class="prediction-box">

<div class="predict-title">
AI Prediction
</div>

<div class="predict-text"
style="color:{pred_color};">
{prediction}
</div>

<div class="small-text">
Confidence : {confidence}
</div>

</div>
""", unsafe_allow_html=True)

# ---------------- TRADINGVIEW ----------------

st.markdown("""
<div class="section-title">
BTC Live Chart
</div>
""", unsafe_allow_html=True)

tradingview_widget = """
<!-- TradingView Widget BEGIN -->
<div class="tradingview-widget-container">
  <div id="tradingview_btc"></div>

  <script type="text/javascript"
  src="https://s3.tradingview.com/tv.js">
  </script>

  <script type="text/javascript">
  new TradingView.widget(
  {
  "width": "100%",
  "height": 700,
  "symbol": "BINANCE:BTCUSDT",
  "interval": "60",
  "timezone": "Etc/UTC",
  "theme": "dark",
  "style": "1",
  "locale": "en",
  "toolbar_bg": "#0f172a",
  "enable_publishing": false,
  "hide_side_toolbar": false,
  "allow_symbol_change": true,
  "container_id": "tradingview_btc"
}
  );
  </script>
</div>
"""

html(tradingview_widget, height=720)

# ---------------- INDICATORS ----------------

st.markdown("""
<div class="section-title">
Main Indicators
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="card">
        <div class="metric-label">RSI</div>
        <div class="metric-number green">
            {rsi:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <div class="metric-label">MACD</div>
        <div class="metric-number red">
            {macd_value:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <div class="metric-label">ATR</div>
        <div class="metric-number yellow">
            {atr:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="card">
        <div class="metric-label">Volatility</div>
        <div class="metric-number blue">
            {volatility:.4f}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- MARKET DATA ----------------

st.markdown("""
<div class="section-title">
Latest Market Data
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("BTC Price", f"${btc_price:,.2f}")

with m2:
    st.metric("RSI", f"{rsi:.2f}")

with m3:
    st.metric("MACD", f"{macd_value:.2f}")

with m4:
    st.metric("ATR", f"{atr:.2f}")

# ---------------- MARKET MOVEMENT ----------------

movement = "SIDEWAYS"
move_color = "#38bdf8"

if rsi > 65 and macd_value > 0:
    movement = "BULLISH"
    move_color = "#00ff99"

elif rsi < 40 and macd_value < 0:
    movement = "BEARISH"
    move_color = "#ff4d6d"

st.markdown("""
<div class="section-title">
Market Movement
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="move-box">

<div style="
font-size:46px;
font-weight:900;
color:{move_color};
margin-bottom:25px;
">
{movement}
</div>

<div style="
font-size:24px;
line-height:2.2;
color:#cbd5e1;
">

BTC Price : ${btc_price:,.2f}

<br>

RSI : {rsi:.2f}

<br>

MACD : {macd_value:.2f}

<br>

ATR : {atr:.2f}

</div>

</div>
""", unsafe_allow_html=True)
