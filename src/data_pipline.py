import yfinance as yf
import pandas as pd
import numpy as np

def fetch_and_clean_data(tickers: list, start_date: str, end_date: str) -> tuple:
    """
    Downloads adjusted close prices, calculates log returns, and clips outliers.
    Returns: (stock_returns, spy_returns)
    """
    tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', # Tech Sector
    'JPM', 'BAC', 'GS', 'MS',                              # Financials Sector
    'JNJ', 'UNH', 'PFE', 'ABBV',                           # Healthcare Sector
    'XOM', 'CVX',                                          # Energy Sector
    'WMT', 'PG', 'KO', 'PEP',                              # Consumer Def Sector
    'CAT', 'DE', 'BA',                                     # Industrials Sector
    'SPY'                                                  # Benchmark
    ]
    data = yf.download(tickers, start="2014-01-01", end="2024-01-01")["Close"]

    # Calculate log returns and separate benchmark
    log_returns = np.log(data / data.shift(1)).dropna()
    spy_returns = log_returns['SPY']
    stock_returns = log_returns.drop(columns=['SPY'])
    stock_returns = stock_returns.clip(lower=stock_returns.quantile(0.01), upper=stock_returns.quantile(0.99), axis=1)
        
    return stock_returns, spy_returns