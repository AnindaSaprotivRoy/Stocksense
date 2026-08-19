# analysis.py — deterministic, inspectable signal scoring and verdict synthesis.
#
# This module is PURE: no network, no Streamlit, no LLM. Every number the
# verdict shows is produced here, by arithmetic you can read top to bottom.
# If you want to change the methodology, this is the only file to edit.
#
# Contract for every signal:
#   * normalise to a score in [-1, +1]  (+1 = maximally bullish)
#   * carry a fixed weight from SIGNAL_WEIGHTS
#   * declare whether its input data was actually available
#   * carry a human-readable `evidence` string stating the raw numbers used
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tunables — the entire methodology's free parameters, in one place.
# ---------------------------------------------------------------------------

SIGNAL_WEIGHTS = {
    "trend":     0.25,   # SMA20 vs SMA50 structure + price position
    "forecast":  0.25,   # Prophet expected return, discounted by its own interval
    "sentiment": 0.20,   # recency-decayed VADER on news headlines
    "momentum":  0.15,   # RSI-14, read as mean-reversion
    "volume":    0.15,   # volume trend as directional confirmation
}

# Verdict thresholds on the composite score.
BUY_THRESHOLD = 0.25
SELL_THRESHOLD = -0.25

# Confidence banding.
CONF_LOW_MAX = 0.40
CONF_MODERATE_MAX = 0.70

# Saturation constants: the raw value at which a signal reaches +/-1.
TREND_SPREAD_SAT = 0.05     # 5% SMA20-vs-SMA50 separation saturates
TREND_POSITION_SAT = 0.08   # 8% price-vs-SMA50 distance saturates
RSI_SAT = 20.0              # RSI 30 -> +1, RSI 70 -> -1
SENTIMENT_SAT = 0.35        # decayed VADER compound of 0.35 saturates
VOLUME_SAT = 0.50           # 50% above the 20d volume baseline saturates
DISPERSION_SAT = 0.80       # weighted std at which agreement hits zero

# Horizon-dependent expected-return saturation for the forecast signal.
RETURN_SAT = {"Next Hour": 0.005, "Next Day": 0.02}

# Data-thinness caps.
MIN_ARTICLES_FOR_FULL_SENTIMENT = 3
MIN_ARTICLES_FOR_FULL_CONFIDENCE = 3
MIN_BARS_FOR_FULL_CONFIDENCE = 60
THIN_SENTIMENT_PENALTY = 0.6

DISCLAIMER = (
    "Automated signal aggregation, not financial advice. This verdict is a weighted "
    "average of the indicators shown above; it has not been backtested and carries no "
    "demonstrated predictive validity. Do not trade on it."
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    key: str
    label: str
    score: float          # [-1, +1]; 0.0 when unavailable
    available: bool
    evidence: str         # the raw numbers this score came from
    caveat: str = ""      # known weakness of this signal, surfaced in the UI

    @property
    def weight(self) -> float:
        return SIGNAL_WEIGHTS[self.key]


@dataclass
class Verdict:
    action: str                       # BUY / HOLD / SELL
    composite: float                  # [-1, +1]
    confidence: float                 # [0, 1]
    confidence_label: str             # LOW / MODERATE / HIGH
    coverage: float
    agreement: float
    magnitude: float
    signals: list = field(default_factory=list)          # all Signals, incl. unavailable
    contributions: dict = field(default_factory=dict)    # key -> contribution (sums to composite)
    drivers: list = field(default_factory=list)          # (Signal, contribution), positive, desc
    drags: list = field(default_factory=list)            # (Signal, contribution), negative, asc
    conflicts: list = field(default_factory=list)        # human-readable conflict statements
    caps_applied: list = field(default_factory=list)     # why confidence was capped
    disclaimer: str = DISCLAIMER


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    return float(min(hi, max(lo, x)))


def _finite(x) -> Optional[float]:
    """Coerce to float, or None if missing/non-finite."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. Returns a series aligned to `close`."""
    close = pd.to_numeric(close, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing == EWM with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # All-gain windows -> RSI 100; all-loss windows -> RSI 0.
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    return out


def compute_indicators(daily: pd.DataFrame) -> dict:
    """
    Compute technical indicators from a daily OHLCV frame.

    `daily` must have columns: ds, close, volume (volume may be all-NaN).
    Every key may be None when there aren't enough bars to compute it.
    """
    out = {
        "bars": 0, "last_close": None, "sma20": None, "sma50": None,
        "rsi14": None, "vol_5d": None, "vol_20d": None, "vol_ratio": None,
        "chg_5d_pct": None, "sma20_series": None, "sma50_series": None,
        "rsi_series": None,
    }
    if daily is None or daily.empty or "close" not in daily.columns:
        return out

    close = pd.to_numeric(daily["close"], errors="coerce").dropna()
    if close.empty:
        return out

    out["bars"] = int(len(close))
    out["last_close"] = _finite(close.iloc[-1])

    if len(close) >= 20:
        s20 = close.rolling(20).mean()
        out["sma20"] = _finite(s20.iloc[-1])
        out["sma20_series"] = s20
    if len(close) >= 50:
        s50 = close.rolling(50).mean()
        out["sma50"] = _finite(s50.iloc[-1])
        out["sma50_series"] = s50
    if len(close) >= 15:
        r = rsi(close)
        out["rsi14"] = _finite(r.iloc[-1])
        out["rsi_series"] = r
    if len(close) >= 6:
        prev = _finite(close.iloc[-6])
        last = out["last_close"]
        if prev and last and prev != 0:
            out["chg_5d_pct"] = (last - prev) / prev * 100.0

    if "volume" in daily.columns:
        vol = pd.to_numeric(daily["volume"], errors="coerce").dropna()
        if len(vol) >= 20:
            v5 = _finite(vol.iloc[-5:].mean())
            v20 = _finite(vol.iloc[-20:].mean())
            out["vol_5d"], out["vol_20d"] = v5, v20
            if v5 is not None and v20:
                out["vol_ratio"] = v5 / v20

    return out


def summarize_forecast(forecast: pd.DataFrame, last_price: float) -> Optional[dict]:
    """
    Reduce a Prophet forecast frame to the two numbers the verdict needs:
    expected return at the final horizon step, and the half-width of the
    prediction interval there (as a fraction of price).
    """
    if forecast is None or forecast.empty or not last_price:
        return None
    needed = {"yhat", "yhat_lower", "yhat_upper"}
    if not needed.issubset(forecast.columns):
        return None

    row = forecast.iloc[-1]
    yhat = _finite(row["yhat"])
    lo = _finite(row["yhat_lower"])
    hi = _finite(row["yhat_upper"])
    if yhat is None:
        return None

    exp_ret = (yhat - last_price) / last_price
    band = None
    if lo is not None and hi is not None:
        band = ((hi - lo) / 2.0) / last_price

    return {
        "yhat": yhat,
        "expected_return": exp_ret,
        "band_halfwidth_pct": band,
        "target_ts": row.get("ds"),
    }


# ---------------------------------------------------------------------------
# Signal scorers — one per row of SIGNAL_WEIGHTS
# ---------------------------------------------------------------------------

def score_trend(ind: dict) -> Signal:
    """
    Moving-average structure. Two components:
      60%  SMA20 vs SMA50 separation  (is the short trend above the long one?)
      40%  price vs SMA50 position    (is price extended above/below the base?)
    """
    sma20, sma50, price = ind.get("sma20"), ind.get("sma50"), ind.get("last_close")
    if not sma20 or not sma50 or not price:
        return Signal("trend", "Trend (MA structure)", 0.0, False,
                      "Needs 50 daily bars for SMA20/SMA50; not enough history.")

    spread = (sma20 - sma50) / sma50
    position = (price - sma50) / sma50
    score = 0.6 * _clip(spread / TREND_SPREAD_SAT) + 0.4 * _clip(position / TREND_POSITION_SAT)

    cross = "above" if spread > 0 else "below"
    ev = (f"SMA20 {sma20:,.2f} is {abs(spread) * 100:.2f}% {cross} SMA50 {sma50:,.2f}; "
          f"price {price:,.2f} sits {position * 100:+.2f}% vs SMA50.")
    return Signal("trend", "Trend (MA structure)", _clip(score), True, ev)


def score_momentum(ind: dict) -> Signal:
    """
    RSI-14, read as MEAN REVERSION: overbought is bearish, oversold is bullish.
    RSI 70 -> -1, RSI 50 -> 0, RSI 30 -> +1.

    This deliberately opposes `trend` during strong moves. That is not a bug:
    a stock ripping upward genuinely is both trending and overbought, and the
    verdict surfaces that as a conflict rather than hiding it.
    """
    r = ind.get("rsi14")
    if r is None:
        return Signal("momentum", "Momentum (RSI-14)", 0.0, False,
                      "Needs 15 daily bars for RSI-14; not enough history.")

    score = _clip((50.0 - r) / RSI_SAT)
    if r >= 70:
        read = "overbought — bearish on a mean-reversion reading"
    elif r <= 30:
        read = "oversold — bullish on a mean-reversion reading"
    else:
        read = "neutral zone"
    ev = f"RSI-14 = {r:.1f} ({read})."
    return Signal("momentum", "Momentum (RSI-14)", score, True, ev,
                  caveat="Scored as mean-reversion, so it will oppose the trend signal "
                         "in a strong directional move.")


def score_sentiment(scalar: Optional[float], n_articles: int) -> Signal:
    """
    Recency-decayed VADER compound from signals.recent_sentiment_scalar().
    Damped when the sample is thin.
    """
    if scalar is None or n_articles == 0:
        return Signal("sentiment", "News sentiment", 0.0, False,
                      "No headlines retrieved (missing NEWS_API_KEY, or no coverage).")

    score = _clip(scalar / SENTIMENT_SAT)
    note = ""
    if n_articles < MIN_ARTICLES_FOR_FULL_SENTIMENT:
        score *= THIN_SENTIMENT_PENALTY
        note = f" Damped x{THIN_SENTIMENT_PENALTY} — only {n_articles} article(s)."

    ev = (f"Recency-weighted VADER compound = {scalar:+.3f} across {n_articles} headline(s), "
          f"180-minute half-life.{note}")
    return Signal("sentiment", "News sentiment", score, True, ev,
                  caveat="VADER is a general-purpose social-media lexicon, not a financial one. "
                         "It misreads domain phrasing (e.g. 'beats earnings but guides lower', "
                         "'recalls 500,000 vehicles'). Treat as weak evidence.")


def score_forecast(fc: Optional[dict], horizon: str) -> Signal:
    """
    Prophet expected return, normalised by a horizon-appropriate scale, then
    DISCOUNTED BY ITS OWN UNCERTAINTY: if the prediction interval is wide
    relative to the move being predicted, the signal shrinks toward zero.
    """
    if not fc or fc.get("expected_return") is None:
        return Signal("forecast", "Prophet forecast", 0.0, False,
                      "Forecast unavailable (insufficient price history to fit).")

    ret = fc["expected_return"]
    sat = RETURN_SAT.get(horizon, 0.02)
    raw = _clip(ret / sat)

    band = fc.get("band_halfwidth_pct")
    if band and band > 0:
        shrink = _clip(abs(ret) / band, 0.0, 1.0)
    else:
        shrink = 1.0
    score = raw * shrink

    ev = (f"Prophet expects {ret * 100:+.2f}% over the {horizon.lower()} horizon "
          f"(raw signal {raw:+.2f}).")
    if band:
        ev += (f" Prediction interval half-width is {band * 100:.2f}% of price, so the signal "
               f"is discounted x{shrink:.2f} to {score:+.2f}.")
    return Signal("forecast", "Prophet forecast", score, True, ev,
                  caveat="Prophet is built for daily/weekly series with genuine seasonality. "
                         "On short-horizon price data it largely extrapolates recent drift; "
                         "the uncertainty discount above is a partial correction, not a fix.")


def score_volume(ind: dict) -> Signal:
    """
    Volume has NO direction of its own — rising volume is bullish on a rising
    price and bearish on a falling one. So it is scored as:

        sign(5-day price change) x confirmation strength

    Flat volume scores ~0 rather than voting either way.
    """
    ratio, chg = ind.get("vol_ratio"), ind.get("chg_5d_pct")
    if ratio is None or chg is None:
        return Signal("volume", "Volume trend", 0.0, False,
                      "Needs 20 daily bars of volume; not enough history.")

    strength = _clip((ratio - 1.0) / VOLUME_SAT, 0.0, 1.0)
    direction = 1.0 if chg > 0 else (-1.0 if chg < 0 else 0.0)
    score = direction * strength

    if strength < 0.1:
        read = "volume is near its 20-day baseline — no confirmation either way"
    elif direction > 0:
        read = "elevated volume confirming a rising price — bullish"
    elif direction < 0:
        read = "elevated volume confirming a falling price — bearish"
    else:
        read = "price flat over 5 days — no direction to confirm"

    ev = (f"5-day average volume is {ratio:.2f}x the 20-day average, with price "
          f"{chg:+.2f}% over 5 days: {read}.")
    return Signal("volume", "Volume trend", score, True, ev)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def build_verdict(signals: list, news_count: int = 0, daily_bars: int = 0) -> Verdict:
    """
    Combine scored signals into a verdict.

        composite      = SUM(w_i * s_i) / SUM(w_i)   over AVAILABLE signals only
        contribution_i = w_i * s_i / SUM(w_available)

    Contributions sum exactly to the composite, so the contribution chart the UI
    renders IS the arithmetic rather than an illustration of it.

    Renormalising over available signals means missing data does not silently
    drag the score toward neutral — it reduces confidence instead.
    """
    available = [s for s in signals if s.available]
    w_all = sum(SIGNAL_WEIGHTS.values())
    w_avail = sum(s.weight for s in available)

    if w_avail == 0:
        return Verdict(
            action="HOLD", composite=0.0, confidence=0.0, confidence_label="LOW",
            coverage=0.0, agreement=0.0, magnitude=0.0, signals=signals,
            caps_applied=["No signal had usable data — nothing to aggregate."],
        )

    composite = sum(s.weight * s.score for s in available) / w_avail
    contributions = {s.key: (s.weight * s.score) / w_avail for s in available}

    # --- verdict band ---
    if composite >= BUY_THRESHOLD:
        action = "BUY"
    elif composite <= SELL_THRESHOLD:
        action = "SELL"
    else:
        action = "HOLD"

    # --- confidence: coverage x agreement x magnitude ---
    coverage = w_avail / w_all

    # Weighted variance of the signal scores about the composite (which is
    # their weighted mean), so this is a true weighted dispersion.
    wvar = sum(s.weight * (s.score - composite) ** 2 for s in available) / w_avail
    wstd = float(np.sqrt(wvar))
    agreement = _clip(1.0 - wstd / DISPERSION_SAT, 0.0, 1.0)

    # A composite near zero is a weak call even when everything agrees —
    # agreement about nothing should not read as high confidence.
    magnitude = _clip(abs(composite) / 0.5, 0.0, 1.0)

    confidence = float(np.sqrt(coverage)) * agreement * (0.35 + 0.65 * magnitude)

    # --- honest caps on thin data ---
    caps = []
    if coverage < 0.5:
        confidence = min(confidence, CONF_LOW_MAX - 0.01)
        caps.append(f"Capped to LOW: only {coverage * 100:.0f}% of signal weight had data.")
    if news_count < MIN_ARTICLES_FOR_FULL_CONFIDENCE:
        confidence = min(confidence, CONF_MODERATE_MAX - 0.01)
        caps.append(f"Capped to MODERATE: only {news_count} headline(s) available "
                    f"(need {MIN_ARTICLES_FOR_FULL_CONFIDENCE}).")
    if daily_bars < MIN_BARS_FOR_FULL_CONFIDENCE:
        confidence = min(confidence, CONF_MODERATE_MAX - 0.01)
        caps.append(f"Capped to MODERATE: only {daily_bars} daily bars of history "
                    f"(need {MIN_BARS_FOR_FULL_CONFIDENCE}).")

    confidence = _clip(confidence, 0.0, 1.0)
    if confidence < CONF_LOW_MAX:
        conf_label = "LOW"
    elif confidence < CONF_MODERATE_MAX:
        conf_label = "MODERATE"
    else:
        conf_label = "HIGH"

    # --- drivers, drags, conflicts ---
    by_key = {s.key: s for s in available}
    ranked = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    drivers = [(by_key[k], v) for k, v in ranked if v > 0]
    drags = [(by_key[k], v) for k, v in reversed(ranked) if v < 0]

    conflicts = []
    if drivers and drags:
        top_up, up_val = drivers[0]
        top_dn, dn_val = drags[0]
        if up_val > 0.10 and abs(dn_val) > 0.10:
            conflicts.append(
                f"{top_up.label} is pushing bullish ({up_val:+.3f}) while {top_dn.label} is "
                f"pushing bearish ({dn_val:+.3f}). The {action} call is a weighted average of "
                f"genuinely opposed evidence, not a consensus."
            )
    if len(drivers) and len(drags) and agreement < 0.5:
        conflicts.append(
            f"Signal dispersion is high (weighted std {wstd:.2f}); the individual signals "
            f"disagree more than they agree, which is why confidence is held down."
        )

    return Verdict(
        action=action, composite=composite, confidence=confidence,
        confidence_label=conf_label, coverage=coverage, agreement=agreement,
        magnitude=magnitude, signals=signals, contributions=contributions,
        drivers=drivers, drags=drags, conflicts=conflicts, caps_applied=caps,
    )
