import requests
import pandas as pd
from datetime import datetime


def get_data(ticker, start, end, interval):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end, "%Y-%m-%d").timestamp())

    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": interval,
        "events": "div,splits",
        "includePrePost": "false"
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quotes = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        'open': quotes['open'],
        'high': quotes['high'],
        'low': quotes['low'],
        'close': quotes['close'],
        'volume': quotes['volume'],
        'time': timestamps
    })
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df
