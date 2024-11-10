# AI Stock Advisor with Ollama LLM Integration (llama3.2)

This application uses a Large Language Model (LLM) via the Ollama API to generate natural language insights from the technical analysis of stock market data. It is packaged using Docker for efficient deployment and execution.

# About LLMs and Ollama

Large Language Models (LLMs) are advanced machine learning models trained on vast amounts of textual data to understand, generate, and interpret natural language. This can power various applications with Generative AI, from chatbots to automated content creation. Ollama is an innovative platform for deploying and managing LLMs locally, providing the tools to interact with models like llama3.2, which powers this application's predictions and insights.

# Getting Started

Step 1: Download the *[Ollama Docker image](https://hub.docker.com/r/ollama/ollama)* from Docker Hub:

Step 2: Deploy Ollama in a Docker container:
```
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

Step 3: Install the *[LLM Model](https://ollama.com/library)* llama3.2 model:
```
docker exec -it ollama ollama run llama3.2
```

Step 4: Confirm that the llama3.2:latest model is installed:
```
docker exec -it ollama ollama list
```

<img src="./Img/llama3.2.png"> 

Step 5: Access the running service at *[http://localhost:11434](http://localhost:11434/)* to confirm that it's live:

Step 6: Run a test on the model as a service, use the following curl command:
```
curl -X POST http://localhost:11434/api/generate \
     -H "Content-Type: application/json" \
     -d '{"model":"llama3.2:latest", "prompt":"What is LLM?", "stream":false}'
```
<img src="./Img/llama3.2_test.png"> 

Step 7: Sample python file for API Testing - [ollama.py](./ollama.py), to test the ollama API from code:

<img src="./Img/ollama_response.png">

# Extending the Python Application to an AI Stock Advisor

Enhance this idea to create a personal stock advisor application, including steps for setup and deployment in Docker.

How It Works - This application tracks stock data in real-time, calculates technical indicators, and provides predictions using the Ollama LLM. It fetches data at a set interval, performs technical analysis, and generates insights through the ollama API.

<img src="./Img/processflow.png">

Step 8: Set Up application structure:

```bash
stock_tracker/
├── Dockerfile
├── app
  ├── requirements.txt
  ├── main.py
  ├── data_fetcher.py
  ├── analysis.py
  └── llm_insights.py
```

Step 9: Dockerfile for containerizing the Streamlit application:

```
# Use Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY app/requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY app /app

# Expose the Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
```

Step 10: In requirements.txt, include the following dependencies:

|    Python library      | description                    |
|------------------------|-------------------------------|
|`streamlit`             | For creating an interactive web app |
|`yfinance`              | For fetching real-time stock data |
|`numpy and pandas`      | For data manipulation and calculations |
|`matplotlib and plotly` | For data visualization |
|`schedule`              | For scheduling regular updates |
|`ollama`                | For API interaction with Ollama LLM |

```
streamlit
yfinance
numpy
pandas
matplotlib
schedule
plotly
ollama
```
Step 11: Create Python files

`data_fetcher.py`

```
import yfinance as yf
import pandas as pd

def fetch_historical_data(stock_symbol, period="1d", interval="1m"):
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period=period, interval=interval)
    return data
```

`analysis.py`

```
import pandas as pd

def calculate_technical_indicators(stock_data):
    """
    Calculate technical indicators for stock data.
    
    Args:
        stock_data (pd.DataFrame): The stock data with at least 'Close' price column.
        
    Returns:
        pd.DataFrame: Stock data with added EMA and Bollinger Bands.
    """
    # Ensure 'Close' column exists
    if 'Close' not in stock_data.columns:
        raise ValueError("DataFrame must contain 'Close' column for technical indicators.")
    
    # Calculate Exponential Moving Average (EMA)
    stock_data['EMA'] = stock_data['Close'].ewm(span=20, adjust=False).mean()

    # Calculate Bollinger Bands
    stock_data['20 Day MA'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['20 Day STD'] = stock_data['Close'].rolling(window=20).std()
    stock_data['Upper Band'] = stock_data['20 Day MA'] + (stock_data['20 Day STD'] * 2)
    stock_data['Lower Band'] = stock_data['20 Day MA'] - (stock_data['20 Day STD'] * 2)
    
    # Clean up columns for better readability
    stock_data = stock_data[['Close', 'EMA', 'Upper Band', 'Lower Band']].copy()
    
    return stock_data
```

`llm_insights.py`

```
def generate_insights(stock_data, stock_symbol):
    """
    Generate insights based on technical indicators in the stock data.
    
    Args:
        stock_data (pd.DataFrame): The stock data with technical indicators.
        stock_symbol (str): The symbol of the stock.
        
    Returns:
        str: Generated insights based on EMA and Bollinger Bands.
    """
    insights = []
    
    # Check for EMA Trend
    if stock_data['Close'].iloc[-1] > stock_data['EMA'].iloc[-1]:
        insights.append(f"The stock price for {stock_symbol} is currently above the EMA, indicating an upward trend.")
    else:
        insights.append(f"The stock price for {stock_symbol} is currently below the EMA, indicating a potential downward trend.")
    
    # Check for Bollinger Bands Signal
    if stock_data['Close'].iloc[-1] > stock_data['Upper Band'].iloc[-1]:
        insights.append(f"{stock_symbol} has breached the upper Bollinger Band, which may indicate overbought conditions.")
    elif stock_data['Close'].iloc[-1] < stock_data['Lower Band'].iloc[-1]:
        insights.append(f"{stock_symbol} has breached the lower Bollinger Band, which may indicate oversold conditions.")
    else:
        insights.append(f"{stock_symbol} is trading within the Bollinger Bands, showing stable market conditions.")
    
    return "\n".join(insights)
```

`main.py`

```
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

```
Step 11: Deploy the Dockerfile:
```
docker build -t stock-tracker .
```

```
docker run -d -p 8501:8501 --name stock-tracker stock-tracker

```

Step 12: Run the application *[http://localhost:8501](http://localhost:8501)*

<img src="./Img/AI_Stock_Advisor.png"> 

# Improvements and Future Scope

Historical Analysis: Integrate historical data for deeper trend analysis.
Real-Time Alerts: Add real-time alerts for significant stock movements.
Additional Gen AI Applications: Extend the app to include personal AI assistants or decision-making tools using the Ollama model.
