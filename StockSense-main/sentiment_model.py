# sentiment_model.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

analyzer = SentimentIntensityAnalyzer()

def sentiment_summary(news_list: list[dict]) -> dict:
    """
    Takes a list of news articles (each dict with 'title' key)
    and returns average sentiment + label.
    """
    if not news_list:
        return {"average_score": 0.0, "label": "⚠️ No recent news"}

    scores = []
    for article in news_list:
        title = article.get("title", "")
        if title:
            score = analyzer.polarity_scores(title)["compound"]
            scores.append(score)

    if not scores:
        return {"average_score": 0.0, "label": "⚠️ No valid titles"}

    avg_score = float(np.mean(scores))
    if avg_score > 0.1:
        label = "📈 Positive"
    elif avg_score < -0.1:
        label = "📉 Negative"
    else:
        label = "⚖️ Neutral"

    return {"average_score": round(avg_score, 3), "label": label}
