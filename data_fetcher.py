import yfinance as yf
import pandas as pd

def fetch_historical_data(stock_symbol, period="1d", interval="1m"):
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period=period, interval=interval)
    return data
