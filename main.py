import requests
import schedule
import time
import streamlit as st
import yfinance as yf
from datetime import datetime

# Constants
OLLAMA_API_URL = "http://host.docker.internal:11434/api/generate"
HEADERS = {"Content-Type": "application/json"}

# Streamlit Input for Stock Symbols
st.title("AI Stock Advisor")
stock_symbols = st.text_input("Enter stock symbols separated by commas:", "AAPL, GOOG, MSFT").split(", ")
interval_minutes = st.number_input("Set update interval in minutes:", min_value=1, max_value=60, value=15)

# Function to call Ollama API for natural language insights
def call_ollama_api(stock_symbol, prompt):
    try:
        response = requests.post(OLLAMA_API_URL, json={"model": "llama3.2:latest", "prompt": prompt, "stream": False}, headers=HEADERS)
        response.raise_for_status()
        return response.json().get("response", "No prediction available")
    except requests.exceptions.RequestException as e:
        st.error(f"Error contacting Ollama API for {stock_symbol}: {e}")
        return "No prediction available"

# Fetch key data points for prediction (EMA, RSI, Bollinger Bands, MACD)
def fetch_key_data(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period="1d", interval="1m")
    
    # Check if historical data is empty or has insufficient data points
    if hist.empty or len(hist) < 20:  # Ensuring we have at least 20 data points for moving average calculations
        st.warning(f"No sufficient data available for {stock_symbol}. Skipping update.")
        return {
            "EMA": "N/A",
            "RSI": "N/A",
            "Bollinger Bands": ("N/A", "N/A"),
            "MACD": "N/A",
            "MACD Signal": "N/A",
            "Current Price": "N/A"
        }
    
    # Calculate key indicators
    ema = hist['Close'].ewm(span=20).mean().iloc[-1]
    rsi = 100 - (100 / (1 + (hist['Close'].diff().iloc[1:].gt(0).sum() / hist['Close'].diff().iloc[1:].lt(0).sum())))
    upper_band = float(hist['Close'].rolling(window=20).mean().iloc[-1] + 2 * hist['Close'].rolling(window=20).std().iloc[-1])
    lower_band = float(hist['Close'].rolling(window=20).mean().iloc[-1] - 2 * hist['Close'].rolling(window=20).std().iloc[-1])
    current_price = float(hist['Close'].iloc[-1])

    # MACD Calculation
    short_ema = hist['Close'].ewm(span=12, adjust=False).mean()
    long_ema = hist['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_value = float(macd.iloc[-1])
    signal_value = float(signal.iloc[-1])

    return {
        "EMA": round(float(ema), 2),
        "RSI": round(float(rsi), 2),
        "Bollinger Bands": (round(lower_band, 2), round(upper_band, 2)),
        "MACD": round(macd_value, 2),
        "MACD Signal": round(signal_value, 2),
        "Current Price": round(current_price, 2)
    }

# Generate natural language insights based on key data
def get_natural_language_insights(stock_symbol, key_data):
    prompt = f"Provide a brief, contextual prediction for {stock_symbol} based on the following indicators: {key_data}. Keep it short."
    return call_ollama_api(stock_symbol, prompt)

# Process stock update for a single stock
def process_stock_update(stock_symbol):
    st.write(f"Updating insights for {stock_symbol}...")
    key_data = fetch_key_data(stock_symbol)
    prediction = get_natural_language_insights(stock_symbol, key_data)
    display_prediction(stock_symbol, key_data, prediction)


# Display prediction and key data in Streamlit
def display_prediction(stock_symbol, key_data, prediction):
    st.subheader(f"{stock_symbol} Prediction")
    
    # Format key data points as a single, comma-separated line
    key_data_line = ", ".join([f"{k}: {v}" for k, v in key_data.items()])    
    st.write("**Key Data Points:**")
    st.write(key_data_line)  # Display in a single line    
    st.write("**Prediction:**")
    st.write(prediction)

# Schedule job for updating insights on multiple stocks
def scheduled_job():
    for stock_symbol in stock_symbols:
        process_stock_update(stock_symbol)

# Run the scheduler in the background
schedule.every(interval_minutes).minutes.do(scheduled_job)

# Streamlit real-time update loop
while True:
    schedule.run_pending()
    time.sleep(1)
