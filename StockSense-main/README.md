# StockSense

A real-time stock analysis and trading platform with sentiment analysis and AI-powered insights. StockSense combines market data, sentiment analysis, and machine learning to provide comprehensive stock market insights and automated trading signals.

## Features

- Real-time stock price monitoring and analysis
- AI-powered market insights using Google's Gemini
- Technical analysis and price predictions using Prophet
- Automated buy/sell signals based on multiple indicators
- News sentiment analysis using VADER
- Interactive charts and visualizations with Plotly
- Auto-refresh functionality for real-time updates

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/StockSense.git
   cd StockSense
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root with:
   ```
   NEWS_API_KEY=your_newsapi_key
   GOOGLE_API_KEY=your_gemini_api_key
   LIVE_API_KEY=your_twelvedata_key
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Project Structure

- `app.py` - Main Streamlit application and UI
- `fetch_data.py` - Data fetching utilities
- `realtime_data.py` - Real-time stock data integration
- `sentiment_model.py` - Sentiment analysis implementation
- `signals.py` - Trading signals generation
- `gemini_client.py` - Google Gemini AI integration

## Dependencies

Key dependencies include:
- Streamlit for the web interface
- Prophet for time series forecasting
- VADER Sentiment for news analysis
- Google Generative AI (Gemini) for market insights
- YFinance for stock data
- Plotly for interactive visualizations

See `requirements.txt` for the complete list of dependencies.
