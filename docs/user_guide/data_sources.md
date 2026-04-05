
### 5.2 Working with Different Data Sources

#### 5.2.1 Yahoo Finance

```python
import pandas as pd
import yfinance as yf
from dquant.models import VolClustGB, VolClustXGB, VolClustLightGBM

# Download data
df = yf.download("BTC-USD", start="2020-01-01", end="2024-01-01")
df = pd.DataFrame({
    'open': df[('Open', 'BTC-USD')].values,
    'high': df[('High', 'BTC-USD')].values,
    'low': df[('Low', 'BTC-USD')].values,
    'close': df[('Close', 'BTC-USD')].values,
    'volume': df[('Volume', 'BTC-USD')].values
}, index=df.index)

# Train model
model = VolClustXGB({}, early_stopping=True)
features = [
    'TR',
    'returns',
    'abs_returns',
    'gap',
    'body',
    'shadow',
    'close_position',
    'roll_atr_14'
]
model.fit(df, feature_list=features, input_bars=70, horizon=20, trees_count=200, show_results=True)
```

#### 5.2.2 MetaTrader 5

```python
import MetaTrader5 as mt5
import pandas as pd
import datetime as dt
from dquant.models import VolClustGB, VolClustXGB, VolClustLightGBM


symbol = "GOLD"          # which symbol to watch
timeframe = mt5.TIMEFRAME_D1   # M1, M5, M15, H1, D1, etc.
days_back = 3000             # how many days of history to load


# Connect to MT5
if not mt5.initialize():
    print("Failed to connect to MetaTrader5")
    quit()

# Check that symbol is available
if not mt5.symbol_select(symbol, True):
    print(f"Symbol {symbol} not found or not enabled")
    mt5.shutdown()
    quit()

# Calculate dates
to_date = dt.datetime.now() + dt.timedelta(hours=3)
from_date = to_date - dt.timedelta(days=days_back)

# Load bars
rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)

mt5.shutdown()  # terminal no longer needed

if rates is None or len(rates) == 0:
    print("No data!")
    quit()

# Convert to DataFrame
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')  # correct time


df.rename(columns={
    'tick_volume': 'volume'
}, inplace=True)

# Train model
model = VolClustXGB({}, early_stopping=True)
features = [
    'TR',
    'returns',
    'abs_returns',
    'gap',
    'body',
    'shadow',
    'close_position',
    'roll_atr_14'
]
model.fit(df, feature_list=features, input_bars=70, horizon=20, trees_count=200, show_results=True)
```

#### 5.2.3 Other Sources

The library accepts any pandas DataFrame with price columns.
