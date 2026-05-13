import requests
import pandas as pd
from datetime import datetime


def get_data(ticker, start, end, interval):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"} # Заголовок обязателен

    # Параметры для получения данных за последний год с дневным интервалом
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end, "%Y-%m-%d").timestamp())

    params = {
        "period1": start_ts,  # Обратите внимание: параметр называется period1, не start
        "period2": end_ts,  # И period2, не end
        "interval": interval,
        "events": "div,splits",
        "includePrePost": "false"  # Исключаем pre/post market данные
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    # Извлечение нужных данных из ответа
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quotes = result["indicators"]["quote"][0] # Цены открытия, закрытия, минимумы и т.д.
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
