# fetch_data.py
import os
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import requests

def _to_naive(ts: pd.Series) -> pd.Series:
    """Make any timezone-aware timestamps tz-naive for Prophet."""
    ts = pd.to_datetime(ts, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        return ts.dt.tz_convert(None)
    return ts.dt.tz_localize(None) if ts.dt.tz is not None else ts

def get_stock_df(ticker: str, days: int = 7, interval: str = "30m") -> pd.DataFrame:
    """
    Download OHLC for ticker and return a clean dataframe with:
      - ds: tz-naive datetime
      - y : float close price
    """
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["ds", "y"])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([c for c in col if c]) for col in df.columns]

    df = df.reset_index()

    # Normalize datetime column
    time_col = None
    for cand in ("Datetime", "Date", "datetime", "date"):
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        time_col = df.columns[0]

    close_candidates = [c for c in df.columns if "close" in c.lower()]
    if not close_candidates:
        return pd.DataFrame(columns=["ds", "y"])
    close_col = close_candidates[0]

    ds = _to_naive(df[time_col])
    y = pd.to_numeric(df[close_col], errors="coerce")

    clean = pd.DataFrame({"ds": ds, "y": y})
    clean = clean.dropna(subset=["ds", "y"]).sort_values("ds")
    clean = clean[~clean["ds"].duplicated(keep="last")].reset_index(drop=True)
    return clean

def get_news(company: str, api_key: str = None):
    """Return list of {title,url,publishedAt,score} (score filled later)."""
    if api_key is None:
        api_key = os.getenv("NEWS_API_KEY")
    """Return list of {title,url,publishedAt,score} (score filled later)."""
    if not api_key:
        return []
    try:
        url = (
            "https://newsapi.org/v2/everything"
            f"?q={company}+stock+NASDAQ&language=en&sortBy=publishedAt&apiKey={api_key}"
        )
        res = requests.get(url, timeout=15)
        if res.status_code != 200:
            print(f"[NewsAPI Error] HTTP {res.status_code} - {res.text}")
            return []
        data = res.json()
        articles = data.get("articles") or []
        news_list = []
        for a in articles[:5]:
            news_list.append({
                "title": a.get("title", "No title"),
                "url": a.get("url", "#"),
                "publishedAt": a.get("publishedAt", None),  # <-- used for recency weights
                "score": 0.0,                               # filled by VADER in app.py
            })
        return news_list
    except Exception as e:
        print(f"[NewsAPI ❌] Error for {company}: {e}")
        return []

def steps_for_window(window_label: str) -> int:
    """30-minute bars: 2 = 1 hour, 48 = 1 day."""
    return 2 if window_label == "Next Hour" else 48
