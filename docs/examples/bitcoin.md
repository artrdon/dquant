## 7. Usage Examples

### 7.1 Bitcoin Volatility Forecast

```python
import pandas as pd
import yfinance as yf
from dquant.models import VolClustGB, VolClustXGB, VolClustLightGBM


# 1. Load data
df = yf.download("BTC-USD", start="2020-01-01", interval='1d', auto_adjust=True)
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

---