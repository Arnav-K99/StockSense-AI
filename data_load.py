import yfinance as yf
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Fetching stock data...")
def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:

    data = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if data.empty:
        raise ValueError(
            f"No data found for ticker '{ticker}'. "
            f"Check if the ticker symbol is correct."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[required_columns]

    data = data.dropna()
    data = data.sort_index()

    return data


def get_data_summary(data: pd.DataFrame) -> dict:

    summary = {
        'total_rows': len(data),
        'date_range': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}",
        'trading_days': len(data),
        'latest_close': round(data['Close'].iloc[-1], 2),
        'highest_close': round(data['Close'].max(), 2),
        'lowest_close': round(data['Close'].min(), 2),
        'avg_volume': int(data['Volume'].mean()),
    }
    return summary
