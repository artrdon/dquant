## 3. Quick Start

### 3.1 Minimal Example

```python
import pandas as pd
import yfinance as yf
from dquant.models import VolClustXGB 

# 1. Load data
df = yf.download("BTC-USD", start="2020-01-01", interval='1d')
df = pd.DataFrame({
    'open': df[('Open', 'BTC-USD')].values,
    'high': df[('High', 'BTC-USD')].values,
    'low': df[('Low', 'BTC-USD')].values,
    'close': df[('Close', 'BTC-USD')].values,
    'volume': df[('Volume', 'BTC-USD')].values
}, index=df.index)

# 2. Create model
model = VolClustXGB({}, early_stopping=True)

# 3. Train model
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


# 4. Make forecast
rez = model.forecast(df.iloc[-70:].copy(), show=True)
```

### 3.2 Step-by-Step Breakdown

**Step 1: Loading Data**
```python
df = yf.download("BTC-USD", start="2020-01-01", interval='1d')
```
We use Yahoo Finance to get historical data. Then we create a new DataFrame based on our data:
```python
df = pd.DataFrame({
    'open': df[('Open', 'BTC-USD')].values,
    'high': df[('High', 'BTC-USD')].values,
    'low': df[('Low', 'BTC-USD')].values,
    'close': df[('Close', 'BTC-USD')].values,
    'volume': df[('Volume', 'BTC-USD')].values
}, index=df.index)
```

The result is a pandas DataFrame with columns open, high, low, close, volume.

**Step 2: Creating the Model**
```python
model = VolClustXGB({}, early_stopping=True)
```
Initialize the model with default parameters and early stopping.

**Step 3: Training**
```python
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
- `input_bars=70` — how many candles to use as input data
- `horizon=20` — forecast volatility for the next 20 days
- `trees_count=200` — maximum number of trees in gradient boosting
- `show_results=True` — show error plot after training

After training, an error plot for train and validation will automatically appear.

**Step 4: Forecast**
```python
rez = model.forecast(df.iloc[-70:].copy(), show=True)
```
Pass the last 70 candles, get volatility forecast for the next 20 days.

The volatility forecast will be displayed as a graph.
<img src="https://github.com/artrdon/dquant/blob/main/readmeforecast.png?raw=true">
Red shows volatility for previous candles, green shows future volatility.

