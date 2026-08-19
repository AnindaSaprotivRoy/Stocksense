# StockSense

A single-page equity research view built with Streamlit. Enter a ticker and get
one continuous flow — snapshot, news, sentiment, technicals, and a synthesized
BUY/HOLD/SELL verdict that shows its arithmetic instead of asserting a number.

> **⚠️ Not financial advice.** The verdict is an automated weighted average of
> the signals below. **It has never been backtested and has no demonstrated
> predictive validity.** It is a transparent way to read several indicators at
> once, not a recommendation. Do not trade on it.

---

## The page

| # | Section | What it shows |
|---|---------|---------------|
| 01 | **Snapshot** | Last price, day change, previous close, day range, volume vs its 20-day average, 52-week range |
| 02 | **News** | Headlines most recent first, each with its VADER compound score and age |
| 03 | **Sentiment** | Recency-weighted score, per-headline breakdown with decay weights, and the model's known limits |
| 04 | **Technical analysis** | Prophet forecast with prediction interval, SMA20/SMA50, RSI-14, volume trend |
| 05 | **Verdict** | BUY/HOLD/SELL, confidence, and a contribution chart showing exactly which signals moved the score and by how much |

Two appendices sit below the flow: a **peer comparison** (up to 3 tickers,
30-day normalised to base 100) and an optional **Gemini chat** in the sidebar.

### Sidebar controls

- **Ticker** — free text (anything yfinance knows: `AAPL`, `TSLA`, `^GSPC`,
  `BTC-USD`) or a quick-pick list.
- **Forecast horizon** — `Next Hour` fits Prophet on 30-minute intraday bars
  with daily seasonality and predicts 2 steps ahead; `Next Day` fits on the
  daily close series and predicts 1 step. The horizon also changes the
  saturation scale the forecast signal is normalised against (0.5% vs 2%).
- **Auto-refresh (60s)** — refreshes the price snapshot only. The forecast and
  news stay cached; refitting Prophet every minute would burn CPU for a
  forecast that barely moves.
- **Hard refresh** — clears every `st.cache_data` entry.

Cache TTLs: quote 30s · news 10m · intraday 5m · daily 15m · forecast 15m ·
company name 24h.

---

## Scoring methodology

All of it lives in **`analysis.py`**, which is pure — no network, no Streamlit,
no LLM. Every number the verdict displays is produced there, and it is the only
file you need to edit to change the methodology.

Each signal normalises to `[-1, +1]` (+1 = maximally bullish), carries a fixed
weight, declares whether its input data was actually available, and carries a
human-readable `evidence` string naming the raw numbers it used.

| Signal | Weight | Normalisation |
|---|---|---|
| **Trend (MA structure)** | 0.25 | `0.6 × clip((SMA20−SMA50)/SMA50 ÷ 0.05)` + `0.4 × clip((price−SMA50)/SMA50 ÷ 0.08)` |
| **Prophet forecast** | 0.25 | `clip(expected_return ÷ scale)`, then **multiplied by** `clip(|ret| ÷ interval_halfwidth, 0, 1)` — a wide prediction interval shrinks the signal toward zero |
| **News sentiment** | 0.20 | `clip(recency_weighted_VADER ÷ 0.35)`, damped ×0.6 below 3 articles |
| **Momentum (RSI-14)** | 0.15 | `clip((50 − RSI) ÷ 20)` — **mean-reversion**: RSI 70 → −1, RSI 30 → +1 |
| **Volume trend** | 0.15 | `sign(5d price change) × clip((vol5d/vol20d − 1) ÷ 0.5, 0, 1)` |

Weights sum to 1.0 and live in `SIGNAL_WEIGHTS`; every saturation constant is a
named module-level tunable directly above it.

### Aggregation

```
composite       = Σ(wᵢ·sᵢ) / Σ(wᵢ)        over AVAILABLE signals only
contributionᵢ   = wᵢ·sᵢ / Σ(w_available)
```

Contributions sum *exactly* to the composite, so the contribution chart in the
UI **is** the arithmetic rather than an illustration of it. Renormalising over
available signals means missing data does not silently drag the score toward
neutral — it lowers confidence instead.

**Verdict:** `composite ≥ +0.25` → BUY · `≤ −0.25` → SELL · otherwise HOLD.

### Confidence

```
coverage   = Σ(w_available) / Σ(w_all)                  thin data      → low
agreement  = clip(1 − weighted_std(sᵢ) / 0.8, 0, 1)     disagreement   → low
magnitude  = clip(|composite| / 0.5, 0, 1)              near-zero call → low

confidence = √coverage × agreement × (0.35 + 0.65 × magnitude)
```

`< 0.40` LOW · `< 0.70` MODERATE · else HIGH, with hard caps: **MODERATE** when
fewer than 3 headlines or fewer than 60 daily bars, **LOW** when coverage is
below 50%. Every cap that fires is shown to the user with its reason.

`weighted_std` is taken about the composite — which *is* the weighted mean of
the scores — so it is a true weighted dispersion, not an approximation.

When the strongest bullish and strongest bearish contributions both exceed
0.10, the page prints an explicit **"signals conflict"** block naming both
sides instead of presenting the average as a consensus.

### Deliberate design choices

- **RSI is read as mean-reversion**, so it *opposes* the trend signal during
  strong moves. That is intentional: a stock ripping upward genuinely is both
  trending and overbought, and the verdict surfaces the tension rather than
  hiding it. Flip the sign in `score_momentum` if you want trend-confirming
  instead.
- **Volume has no direction of its own.** Rising volume is bullish on a rising
  price and bearish on a falling one, so it is scored as direction ×
  confirmation strength and contributes ~0 when volume is at its baseline.
- **The forecast discounts itself.** Prophet's prediction interval is used to
  shrink its own contribution when it is uncertain.
- **Indicators are computed on daily bars, not the intraday series.** 30-minute
  bars over 7 days give ~90 points, so an SMA-50 would span more than half the
  window and mean nothing.
- **News is filtered to headline matches** (`searchIn=title`). An article that
  merely mentions the company in its body tells us nothing about the stock, and
  it is the *title* that VADER scores downstream.
- **The LLM never touches a score.** Gemini only narrates numbers that
  `analysis.py` already computed, under a prompt that forbids adding facts or
  changing the verdict, and it is opt-in behind a button.

### Known weaknesses (read these)

1. **No backtest.** Nothing here demonstrates that these signals predict
   returns. This is the single biggest caveat and it is not fixable by tuning
   weights.
2. **VADER is the wrong lexicon for finance.** It is a general-purpose
   social-media model: "beats earnings but guides lower" and "recalls 500,000
   vehicles" both read as roughly neutral. FinBERT or Loughran-McDonald would be
   the right replacement. This is why sentiment carries a low weight.
3. **Prophet is being used outside its design envelope.** It targets
   daily/weekly series with real seasonality; on short-horizon price data it
   largely extrapolates recent drift. The uncertainty discount is a partial
   mitigation, not a fix.
4. **News coverage is thin and English-only**, capped by the NewsAPI free tier
   (12 headlines per request).

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` in the project root:

```
NEWS_API_KEY=your_newsapi_key      # optional — without it, sentiment is excluded
GOOGLE_API_KEY=your_gemini_key     # optional — narration and chat only
LIVE_API_KEY=your_twelvedata_key   # optional — live price overlay
```

None of the keys are required to run. Missing ones degrade specific sections
and lower the verdict's confidence rather than breaking the page: without
`NEWS_API_KEY` the sentiment signal is marked unavailable and coverage drops to
80%; without `LIVE_API_KEY` the price comes from yfinance's `fast_info`;
without `GOOGLE_API_KEY` the narration and chat panels explain that they are
disabled.

```bash
streamlit run app.py     # http://localhost:8501
python test_analysis.py  # verify the scoring methodology
```

`test_analysis.py` is dependency-light (numpy/pandas only) and prints a
PASS/FAIL line per property, exiting non-zero on any failure.

---

## Project structure

| File | Role |
|------|------|
| `app.py` | Streamlit UI and the page flow. Fetches and renders; computes no scores. |
| `analysis.py` | **The methodology.** Indicators, signal scoring, verdict and confidence. Pure functions, no I/O. |
| `signals.py` | Exponential recency-decay weighting for sentiment (180-minute half-life) |
| `sentiment_model.py` | VADER scoring of headlines, plus the overall label |
| `fetch_data.py` | yfinance price history and quotes, NewsAPI headlines, query building |
| `realtime_data.py` | TwelveData live price, overlaid on the quote when configured |
| `gemini_client.py` | Gemini access, used only for narration and chat |
| `test_analysis.py` | Self-tests pinning the scoring guarantees |

### Data flow

```
fetch_data.get_daily_df ──► analysis.compute_indicators ──► score_trend
                                                        ├─► score_momentum
                                                        └─► score_volume
fetch_data.get_stock_df ──► Prophet (app.py) ──► analysis.summarize_forecast ──► score_forecast
fetch_data.get_news ──► sentiment_model.score_articles ──► signals.recent_sentiment_scalar ──► score_sentiment

                        all five ──► analysis.build_verdict ──► Verdict(action, composite, confidence, contributions)
```
