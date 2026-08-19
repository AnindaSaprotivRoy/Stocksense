"""
Self-tests for the verdict methodology. Run: python test_analysis.py

These pin the properties the verdict is supposed to guarantee, so a change to
the weights or normalisation can't silently break the arithmetic or the
honesty rules around confidence.
"""
import numpy as np
import pandas as pd

import analysis as A

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def sig(key, label, score, available=True):
    return A.Signal(key, label, score, available, "test")


def all_signals(trend=0.0, forecast=0.0, sentiment=0.0, momentum=0.0, volume=0.0):
    return [sig("trend", "Trend", trend), sig("forecast", "Forecast", forecast),
            sig("sentiment", "Sentiment", sentiment), sig("momentum", "Momentum", momentum),
            sig("volume", "Volume", volume)]


print("\n— weights & thresholds —")
check("weights sum to 1.0", abs(sum(A.SIGNAL_WEIGHTS.values()) - 1.0) < 1e-9)
check("thresholds are symmetric", A.BUY_THRESHOLD == -A.SELL_THRESHOLD)

print("\n— composite arithmetic —")
v = A.build_verdict(all_signals(trend=0.8, forecast=-0.4, sentiment=0.2,
                                momentum=-0.6, volume=0.5), news_count=10, daily_bars=250)
manual = (0.25 * 0.8 + 0.25 * -0.4 + 0.20 * 0.2 + 0.15 * -0.6 + 0.15 * 0.5) / 1.0
check("composite matches hand calculation", abs(v.composite - manual) < 1e-12,
      f"{v.composite} vs {manual}")
check("contributions sum EXACTLY to composite",
      abs(sum(v.contributions.values()) - v.composite) < 1e-12)

print("\n— renormalisation over available signals —")
partial = [sig("trend", "Trend", 1.0), sig("forecast", "Forecast", 0.0, available=False),
           sig("sentiment", "Sentiment", 1.0), sig("momentum", "Momentum", 0.0, available=False),
           sig("volume", "Volume", 0.0, available=False)]
vp = A.build_verdict(partial, news_count=10, daily_bars=250)
check("missing signals do not drag composite toward zero", abs(vp.composite - 1.0) < 1e-12,
      f"got {vp.composite}")
check("coverage reflects only available weight", abs(vp.coverage - 0.45) < 1e-12,
      f"got {vp.coverage}")
check("thin coverage caps confidence to LOW", vp.confidence_label == "LOW",
      f"got {vp.confidence_label} @ {vp.confidence}")
check("cap is explained to the user", any("Capped to LOW" in c for c in vp.caps_applied))

print("\n— verdict banding —")
for score, expected in ((0.9, "BUY"), (0.25, "BUY"), (0.24, "HOLD"), (0.0, "HOLD"),
                        (-0.24, "HOLD"), (-0.25, "SELL"), (-0.9, "SELL")):
    got = A.build_verdict(all_signals(trend=score, forecast=score, sentiment=score,
                                      momentum=score, volume=score), 10, 250).action
    check(f"uniform score {score:+.2f} -> {expected}", got == expected, f"got {got}")

print("\n— confidence honesty —")
agree = A.build_verdict(all_signals(0.8, 0.8, 0.8, 0.8, 0.8), 10, 250)
conflict = A.build_verdict(all_signals(1.0, 1.0, -1.0, -1.0, 0.0), 10, 250)
check("unanimous strong signals -> HIGH confidence", agree.confidence_label == "HIGH",
      f"got {agree.confidence_label} @ {agree.confidence:.3f}")
check("conflicting signals score lower confidence than agreeing ones",
      conflict.confidence < agree.confidence)
check("conflict is reported, not papered over", len(conflict.conflicts) > 0)
check("agreement drops when signals disagree", conflict.agreement < 0.4,
      f"got {conflict.agreement}")

weak = A.build_verdict(all_signals(0.02, 0.02, 0.02, 0.02, 0.02), 10, 250)
check("agreement about ~nothing is not HIGH confidence", weak.confidence_label != "HIGH",
      f"got {weak.confidence_label} @ {weak.confidence:.3f}")

thin_news = A.build_verdict(all_signals(0.9, 0.9, 0.9, 0.9, 0.9), news_count=1, daily_bars=250)
check("few headlines caps confidence below HIGH", thin_news.confidence_label != "HIGH")
short_hist = A.build_verdict(all_signals(0.9, 0.9, 0.9, 0.9, 0.9), news_count=10, daily_bars=20)
check("short history caps confidence below HIGH", short_hist.confidence_label != "HIGH")

print("\n— total data failure —")
none_avail = [sig(k, k, 0.0, available=False) for k in A.SIGNAL_WEIGHTS]
vn = A.build_verdict(none_avail, 0, 0)
check("no data -> HOLD", vn.action == "HOLD")
check("no data -> zero confidence", vn.confidence == 0.0)
check("no data -> explained", len(vn.caps_applied) > 0)

print("\n— drivers / drags —")
vd = A.build_verdict(all_signals(trend=1.0, momentum=-1.0), 10, 250)
check("drivers are positive contributions only", all(c > 0 for _, c in vd.drivers))
check("drags are negative contributions only", all(c < 0 for _, c in vd.drags))
check("drivers sorted strongest first",
      [c for _, c in vd.drivers] == sorted([c for _, c in vd.drivers], reverse=True))

print("\n— indicators —")
up = pd.DataFrame({"ds": pd.date_range("2025-01-01", periods=120),
                   "close": np.linspace(100, 200, 120), "volume": [1e6] * 120})
ind_up = A.compute_indicators(up)
check("monotonic rise -> RSI near 100", ind_up["rsi14"] > 95, f"got {ind_up['rsi14']}")
check("monotonic rise -> SMA20 above SMA50", ind_up["sma20"] > ind_up["sma50"])
check("monotonic rise -> trend signal strongly positive", A.score_trend(ind_up).score > 0.9)
check("RSI is read as mean-reversion (overbought -> bearish)",
      A.score_momentum(ind_up).score < -0.9, f"got {A.score_momentum(ind_up).score}")

down = up.copy()
down["close"] = np.linspace(200, 100, 120)
ind_dn = A.compute_indicators(down)
check("monotonic fall -> RSI near 0", ind_dn["rsi14"] < 5, f"got {ind_dn['rsi14']}")
check("monotonic fall -> trend signal strongly negative", A.score_trend(ind_dn).score < -0.9)

check("empty frame yields no indicators and no crash",
      A.compute_indicators(pd.DataFrame())["bars"] == 0)
check("short history marks trend unavailable",
      not A.score_trend(A.compute_indicators(up.head(10))).available)

print("\n— volume is directional confirmation, not a standalone vote —")
check("baseline volume -> ~no vote",
      abs(A.score_volume({"vol_ratio": 1.0, "chg_5d_pct": 5.0}).score) < 0.05)
check("high volume + rising price -> bullish",
      A.score_volume({"vol_ratio": 1.6, "chg_5d_pct": 5.0}).score > 0.9)
check("high volume + falling price -> bearish",
      A.score_volume({"vol_ratio": 1.6, "chg_5d_pct": -5.0}).score < -0.9)

print("\n— forecast is discounted by its own uncertainty —")
tight = A.score_forecast({"expected_return": 0.02, "band_halfwidth_pct": 0.005}, "Next Day")
wide = A.score_forecast({"expected_return": 0.02, "band_halfwidth_pct": 0.20}, "Next Day")
check("wide prediction interval shrinks the signal", wide.score < tight.score * 0.5,
      f"tight {tight.score:.3f} vs wide {wide.score:.3f}")
check("no forecast -> unavailable", not A.score_forecast(None, "Next Day").available)

print("\n— sentiment —")
check("no news -> unavailable", not A.score_sentiment(None, 0).available)
thin = A.score_sentiment(0.5, 1).score
full = A.score_sentiment(0.5, 8).score
check("thin sample is damped", thin < full, f"{thin} vs {full}")
check("sentiment carries its VADER caveat", "VADER" in A.score_sentiment(0.5, 8).caveat)

print("\n— disclaimer —")
check("every verdict carries a disclaimer", bool(A.build_verdict(all_signals(), 5, 250).disclaimer))
check("disclaimer says not financial advice", "not financial advice" in A.DISCLAIMER.lower())

print("\n" + ("=" * 60))
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    raise SystemExit(1)
print("ALL CHECKS PASSED")
