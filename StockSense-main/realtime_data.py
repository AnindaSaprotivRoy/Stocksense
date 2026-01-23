# realtime_data.py
import os
import requests
from dotenv import load_dotenv

# ✅ Load environment variables from .env
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

LIVE_API_KEY = os.getenv("LIVE_API_KEY")


def get_live_price(symbol: str):
    """
    Fetch the latest stock price for a given symbol using Twelve Data API.
    Automatically uses LIVE_API_KEY from .env.
    Returns a float price or None if it fails.
    """
    if not LIVE_API_KEY:
        print("❌ LIVE_API_KEY missing. Check your .env file.")
        return None

    try:
        # ✅ Use LIVE_API_KEY here (not NEWS_API_KEY)
        url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={LIVE_API_KEY}"
        res = requests.get(url, timeout=10)
        data = res.json()

        # Expected TwelveData response: {"price": "185.65", "symbol": "AAPL"}
        if "price" in data:
            return float(data["price"])
        else:
            print(f"⚠️ Unexpected API response for {symbol}: {data}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error fetching {symbol}: {e}")
        return None

    except Exception as e:
        print(f"❌ Unexpected error fetching {symbol}: {e}")
        return None
