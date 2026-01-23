import os
from dotenv import load_dotenv, dotenv_values

# --- ✅ Load .env manually and inject both keys ---
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")

load_dotenv(dotenv_path, override=True)

if os.path.exists(dotenv_path):
    env_vars = dotenv_values(dotenv_path)
    for key, val in env_vars.items():
        if key not in os.environ or not os.environ[key]:
            os.environ[key] = val

LIVE_API_KEY = os.getenv("LIVE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

print("🔍 ENV CHECK →",
      "LIVE_API_KEY:", LIVE_API_KEY[:6] if LIVE_API_KEY else "❌ Missing",
      "| NEWS_API_KEY:", NEWS_API_KEY[:6] if NEWS_API_KEY else "❌ Missing")


import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 60 seconds
st_autorefresh(interval=60 * 1000, key="data_refresh_timer")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from prophet import Prophet
import plotly.graph_objects as go
import requests
from realtime_data import get_live_price
from gemini_client import ask_gemini, is_ready

# --- PRICE COLOR HELPER ---
def get_price_color(new_price, prev_price):
    if prev_price is None:
        return "#00E5FF"
    elif new_price > prev_price:
        return "#00FF7F"
    elif new_price < prev_price:
        return "#FF4B4B"
    else:
        return "#CCCCCC"

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="StockSense Pro+", page_icon="📊", layout="wide")

# =========================================================
# 🧑‍💼 USER SIGN-IN
# =========================================================
st.markdown("""
<style>
.signin-box {
    background-color: #161a1e;
    padding: 15px 20px;
    border-radius: 10px;
    color: #E7ECEF;
    box-shadow: 0 0 8px rgba(0,229,255,0.3);
}
.signin-btn {
    background-color: #00E5FF;
    color: black;
    border-radius: 8px;
    font-weight: 600;
    padding: 6px 14px;
    border: none;
    cursor: pointer;
}
.signin-btn:hover {
    background-color: #00b3cc;
}
</style>
""", unsafe_allow_html=True)

# Track sign-in state
if "signed_in" not in st.session_state:
    st.session_state.signed_in = False

# Sign-in section (top left)
col1, col2 = st.columns([0.3, 1])
with col1:
    if not st.session_state.signed_in:
        if st.button("🔐 Sign In", key="signin_main"):
            st.session_state.show_login = True
    else:
        st.markdown(f"**👋 Welcome, {st.session_state.get('user_email', 'Trader')}!**")

# Popup modal for login
if st.session_state.get("show_login", False):
    with st.form("login_form"):
        st.subheader("🔑 Sign In to Your Account")
        email = st.text_input("Email or Username")
        password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Sign In")

        if login_btn:
            if email and password:
                st.session_state.signed_in = True
                st.session_state.user_email = email
                st.session_state.show_login = False
                st.success(f"✅ Logged in as {email}")
            else:
                st.error("❌ Please fill in both fields.")

# Divider
st.markdown("---")

# =========================================================
# ⚙️ AUTO BUY/SELL FEATURE (Unified Frontend Mock)
# =========================================================
if st.session_state.signed_in:
    if "auto_trade_enabled" not in st.session_state:
        st.session_state.auto_trade_enabled = False

    st.markdown("""
    <style>
    .trade-btn {
        display: inline-block;
        font-weight: 600;
        color: black;
        padding: 10px 20px;
        border-radius: 10px;
        border: none;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.4);
    }
    .trade-btn:hover {
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

    # Toggle button
    if st.session_state.auto_trade_enabled:
        btn_label = "🔴 Stop Auto Buy/Sell"
        btn_color = "#FF4B4B"
    else:
        btn_label = "🟢 Enable Auto Buy/Sell"
        btn_color = "#00FF7F"

    # Render styled button
    if st.button(btn_label, key="auto_trade_toggle"):
        st.session_state.auto_trade_enabled = not st.session_state.auto_trade_enabled

        # Feedback toast
        if st.session_state.auto_trade_enabled:
            st.success("✅ Auto Buy/Sell Enabled")
        else:
            st.warning("🛑 Auto Buy/Sell Stopped")

# --- API & ANALYZER ---
analyzer = SentimentIntensityAnalyzer()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚙️ Controls")
companies = st.sidebar.multiselect(
    "Select Companies or Indexes",
    [
        "Apple (AAPL)", "Tesla (TSLA)", "NVIDIA (NVDA)",
        "Microsoft (MSFT)", "Amazon (AMZN)", "Meta (META)", "Netflix (NFLX)", "Google (GOOGL)",
        "S&P 500 (^GSPC)", "NASDAQ (^IXIC)", "Dow Jones (^DJI)"
    ],
    default=["Apple (AAPL)"]
)

window = st.sidebar.radio("Prediction Window", ["Next Hour", "Next Day"])

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Stock Comparison")
compare_selection = st.sidebar.multiselect(
    "Select Companies/Indexes to Compare",
    [
        "Apple (AAPL)", "Tesla (TSLA)", "NVIDIA (NVDA)",
        "Microsoft (MSFT)", "Amazon (AMZN)", "Meta (META)",
        "Netflix (NFLX)", "Google (GOOGL)",
        "S&P 500 (^GSPC)", "NASDAQ (^IXIC)", "Dow Jones (^DJI)"
    ],
    default=["Apple (AAPL)", "Tesla (TSLA)"]
)

if st.sidebar.button("🔄 Refresh Now"):
    st.cache_data.clear()
    st.rerun()

# =========================
# 💬 StockSense Chat (Gemini)
# =========================

st.sidebar.markdown("---")
st.sidebar.subheader("💬 StockSense Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hi! Ask me about your selected stocks, news, or predictions."}
    ]

def _build_context():
    ctx_lines = []
    try:
        if companies:
            ctx_lines.append(f"Selected: {', '.join(companies)}")
        ctx_lines.append(f"Window: {window}")
        ctx_lines.append("You are an analyst for short-horizon forecasts. Be concise.")
    except Exception:
        pass
    return "\n".join(ctx_lines)

with st.sidebar.expander("Open Chat", expanded=True):
    for msg in st.session_state.chat_history:
        if msg["role"] == "assistant":
            st.markdown(f"**🤖:** {msg['content']}")
        else:
            st.markdown(f"**🧑:** {msg['content']}")

    with st.form(key="stocksense_chat_form", clear_on_submit=True):
        user_q = st.text_input("Type your question…", placeholder="e.g., What does Tesla’s earnings imply for next week?")
        send = st.form_submit_button("Send")

    if send and user_q.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_q.strip()})
        convo = [{"role": "assistant", "content": f"Context:\n{_build_context()}"}] + st.session_state.chat_history[-6:]

        # --- DYNAMIC CHATBOT BEHAVIOR ---
        try:
            get_news
        except NameError:
            from fetch_data import get_news

        user_latest = convo[-1]["content"] if convo else ""

        known_companies = {
            "Apple": "AAPL", "Tesla": "TSLA", "NVIDIA": "NVDA",
            "Microsoft": "MSFT", "Amazon": "AMZN", "Meta": "META",
            "Netflix": "NFLX", "Google": "GOOGL",
            "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI"
        }

        mentioned = None
        for name in known_companies:
            if name.lower() in user_latest.lower():
                mentioned = name
                break

        if mentioned:
            news_headlines = get_news(mentioned)
            recent_titles = "\n".join(
                [f"- {n['title']}" for n in news_headlines[:3]]
            ) if news_headlines else "No recent headlines available."

            if news_headlines:
                avg_sentiment = np.mean([n["score"] for n in news_headlines])
                if avg_sentiment > 0.1:
                    emoji = "📈"
                elif avg_sentiment < -0.1:
                    emoji = "📉"
                else:
                    emoji = "⚖️"
            else:
                emoji = "❔"

            system_prompt = f"""
            You are StockSense AI — a concise financial analyst.
            The user asked about **{mentioned}**.
            Summarize in 2–3 sentences with recent insights, likely reasons for price
            movement, and key news events — all based on financial reasoning.

            {emoji} Context Headlines for {mentioned}:
            {recent_titles}

            Keep your tone factual and concise.
            """

        else:
            system_prompt = """
            You are StockSense AI — a concise, market-aware assistant.
            The user may ask about any stock, index, or company.
            Respond in 2–3 factual sentences, referencing recent market trends or news if relevant.
            """

        reply = ask_gemini(convo, system_prompt=system_prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

if not is_ready():
    st.sidebar.info("Add GOOGLE_API_KEY to your .env to enable StockSense Chat.")


# --- Helper Maps ---
ticker_map = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "NVIDIA (NVDA)": "NVDA",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
    "Google (GOOGL)": "GOOGL",
    "S&P 500 (^GSPC)": "^GSPC",
    "NASDAQ (^IXIC)": "^IXIC",
    "Dow Jones (^DJI)": "^DJI"
}

# --- STOCK DATA & NEWS ---
@st.cache_data(ttl=60)
def get_stock_data(ticker):
    end = datetime.now()
    start = end - timedelta(days=7)
    df = yf.download(ticker, start=start, end=end, interval="30m", progress=False)
    if df is None or df.empty:
        df = yf.download(ticker, period="5d", interval="1h", progress=False)
    if df.empty:
        return pd.DataFrame(columns=["ds", "y"])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if x]) for col in df.columns]
    df = df.reset_index()
    time_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), df.columns[0])
    close_col = next((c for c in df.columns if "close" in c.lower()), None)
    if not close_col:
        return pd.DataFrame(columns=["ds", "y"])
    df = df.rename(columns={time_col: "ds", close_col: "y"})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce").dt.tz_localize(None)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df.dropna(subset=["ds", "y"]).sort_values("ds")

@st.cache_data(ttl=300)
def get_news(company):
    if not NEWS_API_KEY:
        return []
    try:
        url = f"https://newsapi.org/v2/everything?q={company}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        res = requests.get(url, timeout=10)
        data = res.json()
        articles = data.get("articles", [])[:5]
        return [{"title": a.get("title", "No title"), "url": a.get("url", "#"),
                 "score": analyzer.polarity_scores(a.get("title", ""))["compound"]} for a in articles]
    except:
        return []

# --- MAIN LOOP ---
for company in companies:
    ticker = ticker_map[company]
    st.header(f"📈 {company}")
    df = get_stock_data(ticker)
    news = get_news(company)

    if news:
        avg_sentiment = np.mean([n["score"] for n in news])
        sentiment = "📈 Positive" if avg_sentiment > 0.1 else "📉 Negative" if avg_sentiment < -0.1 else "⚖️ Neutral"
    else:
        sentiment = "No recent news."

    st.markdown(f"**Overall Sentiment:** {sentiment}")
    for n in news:
        color = "green" if n["score"] > 0.1 else "red" if n["score"] < -0.1 else "gray"
        st.markdown(f"- <span style='color:{color}'>{n['title']}</span> ([link]({n['url']}))", unsafe_allow_html=True)

    if not df.empty:
        m = Prophet(daily_seasonality=True)
        m.fit(df)
        future_steps = 2 if window == "Next Hour" else 48
        future = m.make_future_dataframe(periods=future_steps, freq="30min")
        forecast = m.predict(future)
        last_actual_time = df["ds"].max()
        forecast_future = forecast[forecast["ds"] > last_actual_time - timedelta(hours=1)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode="lines", name="Actual", line=dict(color="deepskyblue", width=2)))
        fig.add_trace(go.Scatter(x=forecast_future["ds"], y=forecast_future["yhat"], mode="lines", name="Forecast", line=dict(color="orange", dash="dot", width=2)))
        fig.update_layout(title=f"{company} Price Forecast ({window})", xaxis_title="Time", yaxis_title="Price (USD)", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        if LIVE_API_KEY:
            live = get_live_price(ticker)
            if live:
                if f"last_price_{ticker}" not in st.session_state:
                    st.session_state[f"last_price_{ticker}"] = None
                prev_price = st.session_state[f"last_price_{ticker}"]
                color = get_price_color(live, prev_price)
                st.session_state[f"last_price_{ticker}"] = live
                st.markdown(f"<div style='background-color:#161a1e; padding:16px; border-radius:12px;text-align:center;'><h3 style='color:{color};'>💹 Live Price for {company}: ${live:.2f}</h3></div>", unsafe_allow_html=True)
            else:
                st.warning(f"Live price not available for {company}.")
        else:
            st.info("Set LIVE_API_KEY in .env to enable live prices.")

# Footer
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# =========================================================
# 📊 STOCK COMPARISON (display below forecasts)
# =========================================================
if compare_selection:
    st.markdown("---")
    st.subheader("📊 Stock Comparison")
    st.write("Comparing selected stocks:")
    compare_tickers = [ticker_map[c] for c in compare_selection]

    # Load 30-day daily data for all chosen tickers
    start = datetime.now() - timedelta(days=30)
    end = datetime.now()

    compare_data = {}
    for name, ticker in zip(compare_selection, compare_tickers):
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
        if df.empty:
            continue
        df["Normalized"] = df["Close"] / df["Close"].iloc[0] * 100
        compare_data[name] = df

    if compare_data:
        # --- Line chart: Normalized performance ---
        fig = go.Figure()
        for name, df in compare_data.items():
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["Normalized"],
                mode="lines",
                name=name,
                line=dict(width=2)
            ))
        fig.update_layout(
            title="Normalized Price Comparison (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="Normalized Price (Base = 100)",
            template="plotly_dark",
            legend=dict(orientation="h", x=0, y=1.15),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Table: % Change Summary ---
        perf_data = []
        for name, df in compare_data.items():
            if not df.empty and "Close" in df.columns:
                first_close = float(df["Close"].iloc[0])
                last_close = float(df["Close"].iloc[-1])
                if first_close > 0:
                    change = ((last_close - first_close) / first_close) * 100
                    perf_data.append({"Company": name, "Change (%)": round(change, 2)})

        if perf_data:
            perf_df = pd.DataFrame(perf_data).sort_values(by="Change (%)", ascending=False).reset_index(drop=True)
            st.dataframe(perf_df, use_container_width=True)
        else:
            st.warning("⚠️ Could not compute performance for selected stocks.")
    else:
        st.warning("⚠️ No valid data found for selected stocks.")
else:
    st.info("Select at least two companies or indexes to compare.")
