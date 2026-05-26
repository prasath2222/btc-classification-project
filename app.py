import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="BTC AI Dashboard",
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
}

/* MAIN */

.block-container{
    max-width:1600px;
    padding-top:1rem;
    padding-left:2rem;
    padding-right:2rem;
}

/* TITLES */

.main-title{
    font-size:58px;
    font-weight:800;
    color:white;
}

.sub-title{
    color:#94a3b8;
    margin-bottom:35px;
}

/* CARDS */

.card{
    background:#0f172a;

    border-radius:22px;

    border:1px solid rgba(255,255,255,0.06);

    padding:28px;

    box-shadow:
    0px 4px 20px rgba(0,0,0,0.35);
}

/* PRICE */

.price-label{
    color:#94a3b8;
    font-size:18px;
}

.price-value{
    font-size:64px;
    font-weight:800;
    color:white;
    margin-top:10px;
}

/* CHANGE */

.change-up{
    color:#22c55e;
    font-size:42px;
    font-weight:800;
}

.change-down{
    color:#ef4444;
    font-size:42px;
    font-weight:800;
}

/* PREDICTION */

.prediction-up{
    background:linear-gradient(
    135deg,
    rgba(34,197,94,0.18),
    rgba(6,78,59,0.55)
    );

    border:1px solid rgba(34,197,94,0.35);

    border-radius:20px;

    padding:28px;

    margin-top:25px;
}

.prediction-down{
    background:linear-gradient(
    135deg,
    rgba(239,68,68,0.18),
    rgba(69,10,10,0.55)
    );

    border:1px solid rgba(239,68,68,0.35);

    border-radius:20px;

    padding:28px;

    margin-top:25px;
}

/* SECTION */

.section-title{
    font-size:34px;
    font-weight:800;
    margin-top:40px;
    margin-bottom:18px;
}

/* INDICATOR */

.indicator-card{
    background:#0f172a;

    border-radius:18px;

    border:1px solid rgba(255,255,255,0.06);

    padding:22px;

    text-align:center;
}

.indicator-name{
    color:#94a3b8;
    font-size:16px;
}

.indicator-value{
    font-size:44px;
    font-weight:800;
    margin-top:12px;
}

/* COLORS */

.rsi{
    color:#38bdf8;
}

.macd{
    color:#f43f5e;
}

.atr{
    color:#f59e0b;
}

.vol{
    color:#22c55e;
}

/* SIGNALS */

.signal-bull{
    background:rgba(34,197,94,0.14);

    border-left:6px solid #22c55e;

    padding:18px;

    border-radius:14px;

    margin-top:15px;

    color:#bbf7d0;

    font-weight:700;
}

.signal-bear{
    background:rgba(239,68,68,0.14);

    border-left:6px solid #ef4444;

    padding:18px;

    border-radius:14px;

    margin-top:15px;

    color:#fecaca;

    font-weight:700;
}

.signal-neutral{
    background:rgba(56,189,248,0.14);

    border-left:6px solid #38bdf8;

    padding:18px;

    border-radius:14px;

    margin-top:15px;

    color:#bae6fd;

    font-weight:700;
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
        font-size:30px;
    }

}

</style>

""", unsafe_allow_html=True)

# =====================================================
# TITLES
# =====================================================

st.markdown("""
<div class="main-title">
BTC AI Prediction Dashboard
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-title">
Live Bitcoin AI Analysis • Technical Indicators • TradingView Chart
</div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================

df = yf.download(
    "BTC-USD",
    interval="1h",
    period="60d",
    auto_adjust=True
)

# FIX COLUMN BUG
if isinstance(df.columns, pd.MultiIndex):

    df.columns = df.columns.get_level_values(0)

df = df.reset_index()

# KEEP NEEDED
df = df[
    [
        "Datetime",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]
]

# FORCE FLOAT
for col in [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna()

# =====================================================
# INDICATORS
# =====================================================

df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Close"]
).rsi()

macd = ta.trend.MACD(
    close=df["Close"]
)

df["MACD"] = macd.macd()

df["MACD_SIGNAL"] = macd.macd_signal()

df["ATR"] = ta.volatility.AverageTrueRange(
    high=df["High"],
    low=df["Low"],
    close=df["Close"]
).average_true_range()

df["EMA_20"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=20
).ema_indicator()

df["EMA_50"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=50
).ema_indicator()

df["Returns"] = df["Close"].pct_change()

df["Volatility"] = (
    df["Returns"]
    .rolling(24)
    .std()
)

df = df.dropna()

# =====================================================
# TARGET
# =====================================================

df["Target"] = np.where(
    df["Close"].shift(-1) > df["Close"],
    1,
    0
)

# =====================================================
# FEATURES
# =====================================================

features = [
    "RSI",
    "MACD",
    "MACD_SIGNAL",
    "ATR",
    "EMA_20",
    "EMA_50",
    "Volatility"
]

X = df[features]

y = df["Target"]

# =====================================================
# MODEL
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

model = XGBClassifier()

model.fit(X_train, y_train)

# =====================================================
# PREDICT
# =====================================================

latest = X.iloc[-1:]

prediction = model.predict(latest)[0]

confidence = model.predict_proba(latest)[0].max()

# =====================================================
# LIVE PRICE
# =====================================================

live_price = df["Close"].iloc[-1]

prev_price = df["Close"].iloc[-24]

change = live_price - prev_price

change_percent = (
    change / prev_price
) * 100

# =====================================================
# PRICE SECTION
# =====================================================

col1, col2 = st.columns([2,1])

with col1:

    st.markdown(f"""
    <div class="card">

        <div class="price-label">
        BTC/USD
        </div>

        <div class="price-value">
        ${live_price:,.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

with col2:

    if change >= 0:

        st.markdown(f"""
        <div class="card">

            <div class="price-label">
            24H Change
            </div>

            <br>

            <div class="change-up">
            ▲ {change:.2f}
            </div>

            <br>

            <div class="change-up">
            {change_percent:.2f}%
            </div>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="card">

            <div class="price-label">
            24H Change
            </div>

            <br>

            <div class="change-down">
            ▼ {change:.2f}
            </div>

            <br>

            <div class="change-down">
            {change_percent:.2f}%
            </div>

        </div>
        """, unsafe_allow_html=True)

# =====================================================
# PREDICTION
# =====================================================

if prediction == 1:

    st.markdown(f"""
    <div class="prediction-up">

        <h1>
        BTC may go UP
        </h1>

        <h3>
        Confidence : {confidence*100:.2f}%
        </h3>

    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown(f"""
    <div class="prediction-down">

        <h1>
        BTC may go DOWN
        </h1>

        <h3>
        Confidence : {confidence*100:.2f}%
        </h3>

    </div>
    """, unsafe_allow_html=True)

# =====================================================
# INDICATORS
# =====================================================

st.markdown("""
<div class="section-title">
Main Indicators
</div>
""", unsafe_allow_html=True)

i1, i2, i3, i4 = st.columns(4)

with i1:

    st.markdown(f"""
    <div class="indicator-card">

        <div class="indicator-name">
        RSI
        </div>

        <div class="indicator-value rsi">
        {df["RSI"].iloc[-1]:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

with i2:

    st.markdown(f"""
    <div class="indicator-card">

        <div class="indicator-name">
        MACD
        </div>

        <div class="indicator-value macd">
        {df["MACD"].iloc[-1]:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

with i3:

    st.markdown(f"""
    <div class="indicator-card">

        <div class="indicator-name">
        ATR
        </div>

        <div class="indicator-value atr">
        {df["ATR"].iloc[-1]:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)

with i4:

    st.markdown(f"""
    <div class="indicator-card">

        <div class="indicator-name">
        Volatility
        </div>

        <div class="indicator-value vol">
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
    🔴 RSI Overbought : {rsi:.2f}
    </div>
    """, unsafe_allow_html=True)

elif rsi < 30:

    st.markdown(f"""
    <div class="signal-bull">
    🟢 RSI Oversold : {rsi:.2f}
    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown(f"""
    <div class="signal-neutral">
    🔵 RSI Neutral : {rsi:.2f}
    </div>
    """, unsafe_allow_html=True)

if df["MACD"].iloc[-1] > df["MACD_SIGNAL"].iloc[-1]:

    st.markdown("""
    <div class="signal-bull">
    🟢 MACD Bullish Crossover
    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <div class="signal-bear">
    🔴 MACD Bearish Crossover
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# TRADINGVIEW
# =====================================================

st.markdown("""
<div class="section-title">
BTC Live Chart
</div>
""", unsafe_allow_html=True)

tradingview = """

<div class="tradingview-widget-container">

<div id="tradingview_btc"></div>

<script
type="text/javascript"
src="https://s3.tradingview.com/tv.js">
</script>

<script type="text/javascript">

new TradingView.widget(
{
    "width": "100%",
    "height": 750,

    "symbol": "BINANCE:BTCUSDT",

    "interval": "60",

    "timezone": "Asia/Kolkata",

    "theme": "dark",

    "style": "1",

    "locale": "en",

    "toolbar_bg": "#0f172a",

    "enable_publishing": false,

    "allow_symbol_change": true,

    "save_image": true,

    "hide_side_toolbar": false,

    "withdateranges": true,

    "container_id": "tradingview_btc"
}
);

</script>

</div>

"""

st.components.v1.html(
    tradingview,
    height=760,
    scrolling=False
)

# =====================================================
# EMA CHART
# =====================================================

st.markdown("""
<div class="section-title">
EMA 20 vs EMA 50
</div>
""", unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["Datetime"],
        y=df["EMA_20"],
        name="EMA 20",
        line=dict(
            color="#38bdf8",
            width=3
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=df["Datetime"],
        y=df["EMA_50"],
        name="EMA 50",
        line=dict(
            color="#f43f5e",
            width=3
        )
    )
)

fig.update_layout(

    template="plotly_dark",

    paper_bgcolor="#050816",

    plot_bgcolor="#050816",

    height=500,

    font=dict(color="white"),

    xaxis=dict(
        showgrid=False
    ),

    yaxis=dict(
        gridcolor="rgba(255,255,255,0.06)"
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# LATEST DATA
# =====================================================

st.markdown("""
<div class="section-title">
Latest BTC Data
</div>
""", unsafe_allow_html=True)

latest_df = df[
    [
        "Datetime",
        "Close",
        "RSI",
        "MACD",
        "ATR",
        "EMA_20",
        "EMA_50"
    ]
].tail(10)

st.dataframe(
    latest_df,
    use_container_width=True
)
