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
