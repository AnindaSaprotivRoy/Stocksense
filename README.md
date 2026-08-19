# StockSense

StockSense is a Streamlit equity research dashboard. Enter a ticker to view a market snapshot, recent news, sentiment, technical indicators, and a transparent BUY/HOLD/SELL verdict.

> **Not financial advice.** StockSense is an experimental analysis tool. Its scoring methodology has not been backtested and has no demonstrated predictive validity. Do not make trading decisions based solely on its output.

## Features

- Price snapshot with daily change, range, volume, and 52-week context
- Recent company news with VADER sentiment scores
- Recency-weighted headline sentiment
- Technical indicators including SMA20/SMA50, RSI-14, volume trend, and Prophet forecasts
- Transparent weighted scoring with signal contributions and confidence
- Peer comparison for up to three tickers
- Optional Gemini-generated narration and chat
- Optional TwelveData live-price overlay

## Quick Start

The application source is in `StockSense-main`.

```bash
cd StockSense-main
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

To run the scoring self-tests:

```bash
cd StockSense-main
python test_analysis.py
```

## Optional API Keys

Create `StockSense-main/.env` if you want to enable the optional integrations:

```env
NEWS_API_KEY=your_newsapi_key
GOOGLE_API_KEY=your_gemini_key
LIVE_API_KEY=your_twelvedata_key
```

The application can run without these keys. Missing integrations disable only their related features and reduce the available-data coverage used by the verdict.

## How Scoring Works

The verdict combines five signals, when their data is available:

| Signal | Weight |
|---|---:|
| Trend / moving-average structure | 25% |
| Prophet forecast | 25% |
| News sentiment | 20% |
| RSI momentum | 15% |
| Volume trend | 15% |

The final score is renormalized over available signals. A score of `>= 0.25` produces **BUY**, `<= -0.25` produces **SELL**, and values between those thresholds produce **HOLD**. Confidence also considers data coverage, signal agreement, and score magnitude.

## Project Structure

| File | Purpose |
|---|---|
| [`app.py`](StockSense-main/app.py) | Streamlit interface and page flow |
| [`analysis.py`](StockSense-main/analysis.py) | Indicators, signal scoring, verdict, and confidence |
| [`signals.py`](StockSense-main/signals.py) | Recency weighting for sentiment |
| [`sentiment_model.py`](StockSense-main/sentiment_model.py) | Headline sentiment analysis |
| [`fetch_data.py`](StockSense-main/fetch_data.py) | Market and news data retrieval |
| [`realtime_data.py`](StockSense-main/realtime_data.py) | Optional TwelveData price overlay |
| [`gemini_client.py`](StockSense-main/gemini_client.py) | Optional Gemini narration and chat |
| [`test_analysis.py`](StockSense-main/test_analysis.py) | Scoring methodology self-tests |

For the full methodology, design decisions, known limitations, and data-flow details, see [`StockSense-main/README.md`](StockSense-main/README.md).
