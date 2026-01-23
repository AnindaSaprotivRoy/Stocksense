# signals.py — lightweight, explainable signal transforms
import numpy as np
import pandas as pd
from datetime import timezone

def decay_weights(timestamps: pd.Series, now: pd.Timestamp | None = None, half_life_minutes: int = 180) -> np.ndarray:
    """
    Exponential decay weights in minutes. More recent = heavier.
    """
    now = now or pd.Timestamp.now(tz=timezone.utc)
    t = pd.to_datetime(timestamps, utc=True, errors="coerce")
    age_min = (now - t).dt.total_seconds() / 60.0
    lam = np.log(2) / max(1, half_life_minutes)
    w = np.exp(-lam * np.clip(age_min, 0, None))
    return w.fillna(0).to_numpy()

def recent_sentiment_scalar(news: list[dict], half_life_minutes: int = 180) -> float:
    """
    Returns a single scalar in roughly [-1, +1] weighting sentiment by recency.
    Expects each item to have 'publishedAt' and 'score' (compound).
    """
    if not news:
        return 0.0
    df = pd.DataFrame(news)
    if "publishedAt" not in df:
        # fallback: equal weights
        return float(np.nanmean(pd.to_numeric(df.get("score", 0), errors="coerce")))
    w = decay_weights(df["publishedAt"], half_life_minutes=half_life_minutes)
    scores = pd.to_numeric(df.get("score", 0), errors="coerce").fillna(0).to_numpy()
    if w.sum() == 0:
        return 0.0
    return float((w * scores).sum() / w.sum())
