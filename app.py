import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import ta

# ---------------- PAGE ----------------

st.set_page_config(
    page_title="BTC AI Dashboard",
    layout="wide"
)

# ---------------- STYLE ----------------

st.markdown("""
<style>

html, body, [class*="css"]{
    background-color:#050816;
    color:white;
}

/* remove top padding */
.block-container{
    padding-top:2rem;
    padding-left:3rem;
    padding-right:3rem;
}

/* title */
.main-title{
    font-size:72px;
    font-weight:900;
    color:white;
    margin-bottom:10px;
}

/* subtitle */
.sub-title{
    font-size:22px;
    color:#94a3b8;
    margin-bottom:40px;
}

/* cards */
.card{
    background:#071633;
    padding:30px;
    border-radius:22px;
    border:1px solid #102a5c;
}

/* metric heading */
.metric-label{
    color:#94a3b8;
    font-size:18px;
}

/* metric value */
.metric-number{
    font-size:48px;
    font-weight:900;
    margin-top:15px;
}

/* section title */
.section-title{
    font-size:48px;
    font-weight:900;
    margin-top:50px;
    margin-bottom:25px;
}

/* colors */
.green{
    color:#00ff99;
}

.red{
    color:#ff4d4f;
}

.yellow{
    color:#facc15;
}

.blue{
    color:#38bdf8;
}

/* prediction box */
.predict-box{
    padding:40px;
    border-radius:25px;
    margin-top:30px;
}

/* movement box */
.move-box{
    background:#071633;
    border-radius:25px;
    padding:35px;
    margin-top:30px;
    border:1px solid #102a5c;
}

</style>
""", unsafe_allow_html=True)

# ---------------- DATA ----------------

df = yf.download(
    "BTC-USD",
    period="30d",
    interval="1h"
)

df.columns = df.columns.get_level_values(0)

close = df["Close"]

# RSI
df["RSI"] = ta.momentum.RSIIndicator(
    close=close
).rsi()

# MACD
macd = ta.trend.MACD(close=close)

df["MACD"] = macd.macd()

# ATR
df["ATR"] = ta.volatility.AverageTrueRange(
    high=df["High"],
    low=df["Low"],
    close=df["Close"]
).average_true_range()

# ---------------- VALUES ----------------

btc_price = float(df["Close"].iloc[-1])

change = btc_price - float(df["Close"].iloc[-24])

change_percent = (change / btc_price) * 100

rsi = float(df["RSI"].iloc[-1])

macd_value = float(df["MACD"].iloc[-1])

atr = float(df["ATR"].iloc[-1])

volatility = float(close.pct_change().std())

# ---------------- AI ----------------

prediction = "SIDEWAYS"
confidence = 68
pred_color = "#facc15"
box_bg = "#2b2503"

if rsi > 55 and macd_value > 0:
    prediction = "UP"
    confidence = 74
    pred_color = "#00ff99"
    box_bg = "#032b1f"

elif rsi < 45 and macd_value < 0:
    prediction = "DOWN"
    confidence = 74
    pred_color = "#ff4d4f"
    box_bg = "#2b0303"

# ---------------- TITLE ----------------

st.markdown("""
<div class="main-title">
BTC AI Prediction Dashboard
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-title">
Live Bitcoin Analysis • Trading Setup • Technical Indicators
</div>
""", unsafe_allow_html=True)

# ---------------- TOP CARDS ----------------

col1, col2 = st.columns([2,1])

with col1:

    st.markdown(f"""
    <div class="card">

        <div class="metric-label">
            BTC/USD
        </div>

        <div class="metric-number">
            ${btc_price:,.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

with col2:

    color_class = "green" if change > 0 else "red"

    arrow = "▲" if change > 0 else "▼"

    st.markdown(f"""
    <div class="card">

        <div class="metric-label">
            24H Change
        </div>

        <div class="metric-number {color_class}">
            {arrow} {change:,.2f}
        </div>

        <div style="
            margin-top:12px;
            font-size:28px;
            font-weight:700;
        ">
            {change_percent:.2f}%
        </div>

    </div>
    """, unsafe_allow_html=True)

# ---------------- AI PREDICTION ----------------

st.markdown(f"""
<div class="predict-box"
style="
    background:{box_bg};
    border:2px solid {pred_color};
">

    <div style="
        font-size:26px;
        color:white;
    ">
        AI Prediction
    </div>

    <div style="
        font-size:64px;
        font-weight:900;
        margin-top:20px;
        color:{pred_color};
    ">
        BTC may go {prediction}
    </div>

    <div style="
        font-size:30px;
        margin-top:20px;
        color:white;
    ">
        Confidence : {confidence}%
    </div>

</div>
""", unsafe_allow_html=True)

# ---------------- CHART TITLE ----------------

st.markdown("""
<div class="section-title">
BTC Live Chart
</div>
""", unsafe_allow_html=True)

# ---------------- TRADINGVIEW ----------------

tradingview_html = """

<div class="tradingview-widget-container">

  <div id="tradingview_chart"></div>

  <script
    type="text/javascript"
    src="https://s3.tradingview.com/tv.js">
  </script>

  <script type="text/javascript">

  new TradingView.widget(
  {
    "width":"100%",
    "height":700,
    "symbol":"BINANCE:BTCUSDT",
    "interval":"60",
    "timezone":"Etc/UTC",
    "theme":"dark",
    "style":"1",
    "locale":"en",
    "toolbar_bg":"#050816",
    "enable_publishing":false,
    "hide_top_toolbar":false,
    "allow_symbol_change":true,
    "container_id":"tradingview_chart"
  });

  </script>

</div>

"""

components.html(
    tradingview_html,
    height=700
)

# ---------------- INDICATORS ----------------

st.markdown("""
<div class="section-title">
Main Indicators
</div>
""", unsafe_allow_html=True)

a, b, c, d = st.columns(4)

with a:

    st.markdown(f"""
    <div class="card">

        <div class="metric-label">
            RSI
        </div>

        <div class="metric-number green">
            {rsi:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

with b:

    st.markdown(f"""
    <div class="card">

        <div class="metric-label">
            MACD
        </div>

        <div class="metric-number red">
            {macd_value:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

with c:

    st.markdown(f"""
    <div class="card">

        <div class="metric-label">
            ATR
        </div>

        <div class="metric-number yellow">
            {atr:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

with d:

    st.markdown(f"""
    <div class="card">

        <div class="metric-label">
            Volatility
        </div>

        <div class="metric-number blue">
            {volatility:.4f}
        </div>

    </div>
    """, unsafe_allow_html=True)

# ---------------- LATEST MARKET DATA ----------------

st.markdown("""
<div class="section-title">
Latest Market Data
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

m1.metric("BTC Price", f"${btc_price:,.2f}")
m2.metric("RSI", f"{rsi:.2f}")
m3.metric("MACD", f"{macd_value:.2f}")
m4.metric("ATR", f"{atr:.2f}")

# ---------------- MARKET MOVEMENT ----------------

st.markdown("""
<div class="section-title">
Market Movement
</div>
""", unsafe_allow_html=True)

movement = "SIDEWAYS"

if rsi > 55 and macd_value > 0:
    movement = "BULLISH"

elif rsi < 45 and macd_value < 0:
    movement = "BEARISH"

st.markdown(f"""
<div class="move-box">

<div style="
    font-size:40px;
    font-weight:900;
    color:#38bdf8;
">
    {movement}
</div>

<div style="
    margin-top:25px;
    font-size:26px;
    line-height:2;
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
