
## 4. Core Concepts

### 4.1 Input Data

**DataFrame Requirements:**

- Must contain price columns: `open`, `high`, `low`, `close`, `volume` (when using volume features), `time` (when using time features)
- Data must be sorted in ascending time order (from past to future)

**Example of a correct DataFrame:**

| time                | open  | high  | low   | close | volume |
|:--------------------|:------|:------|:------|:------|:-------|
| 2020-01-02 00:00:00 | 100.0 | 102.0 | 99.5  | 101.5 | 1000000|
| 2020-01-03 01:00:00 | 101.5 | 103.0 | 100.5 | 102.0 | 1100000|
| ...                 | ...   | ...   | ...   | ...   | ...    |

### 4.2 Target Variable

The target variable is the **normalized True Range (TR)** for each candle over the forecast horizon.

TR calculation for a single candle:

```
TR = max(high - low, |high - close_prev|, |low - close_prev|) / close
```

where:
- `high - low` — current candle's range
- `|high - close_prev|` — gap from high to previous close
- `|low - close_prev|` — gap from low to previous close
- `close` — current candle's closing price (normalizing factor)

For the first candle in the window where there is no previous close, only the `high - low` range is used.

Normalization by closing price makes TR **scale-invariant**: the value is expressed as a fraction of the current price, allowing correct comparison of volatility at different price levels.

The function returns an array of normalized TR for all candles in the window.

### 4.3 Feature Engineering

You can select the following feature groups for training:

| Feature | Description | Formula |
|---------|------------------------------------------------------------------|---------|
| `TR` | True Range, normalized by closing price | `max(high-low, \|high-prev_close\|, \|low-prev_close\|) / close` |
| `high_low` | High-low range, normalized | `(high - low) / close` |
| `parkinson` | Parkinson volatility | `√((1 / (4 * log(2))) * (log(high / low))²)` |
| `garman_klass` | Garman-Klass volatility | `√(0.5 * log(high/low)² - (2*log(2)-1) * log(close/open)²)` |
| `rogers_satchell` | Rogers-Satchell volatility | `√(log(high/open)*(log(high/open)-log(close/open)) + log(low/open)*(log(low/open)-log(close/open)))` |
| `return` | Logarithmic return | `log(close / prev_close)` |
| `abs_return` | Absolute return value | `\|return\|` |
| `gap` | Gap between open and previous close | `(open - prev_close) / close` |
| `body` | Candle body | `\|close - open\| / close` |
| `shadow` | Total shadow length (upper + lower) | `((high - max(open,close)) + (min(open,close) - low)) / close` |
| `close_position` | Close position within candle range | `(close - low) / (high - low)` |
| `month` | Month (single value, not time series) | `datetime.month` |
| `day_of_month` | Day of month (single value) | `datetime.day` |
| `day_of_week` | Day of week (single value, 1-7) | `datetime.weekday() + 1` |
| `hour` | Hour (single value) | `datetime.hour` |
| `roll_month` | Month (for each candle) | `datetime.dt.month` |
| `roll_day_of_month` | Day of month (for each candle) | `datetime.dt.day` |
| `roll_day_of_week` | Day of week (for each candle, 1-7) | `datetime.dt.weekday` |
| `roll_hour` | Hour (for each candle) | `datetime.dt.hour` |
| `rsi_{window}` | Relative Strength Index (RSI) for the last value | `100 - (100 / (1 + RS))`, where `RS = avg_gain / avg_loss` |
| `atr_{window}` | Average True Range (ATR) for the last value | `average(TR) / close` |
| `bb_{window}` | Bollinger Bands (simplified) for the last value | `(4 * std) / (sma + ε)` |
| `roll_rsi_{window}` | Relative Strength Index (RSI) for each candle | `100 - (100 / (1 + RS))` with rolling window |
| `roll_atr_{window}` | Average True Range (ATR) for each candle | `rolling_mean(TR) / close` |
| `roll_bb_{window}` | Bollinger Bands (simplified) for each candle | `(4 * rolling_std) / (rolling_sma + ε)` |

- `ε` (epsilon) - a small constant to avoid division by zero
- All features with the `roll_` prefix are calculated as a time series (for each candle)
- Features without the `roll_` prefix (except basic `TR`, `return`, `abs_return`, `gap`, `body`, `shadow`, `close_position`) return only the last value
- All price-related features are divided by the current closing price. This makes them scale-invariant and allows the model to work with different instruments and price levels.

### 4.4 Training and Validation

During training, data is automatically split into training and validation sets:

- Default: the last 20% of data is used for validation
- Temporal order is preserved (no random shuffling)
- Error is tracked on the validation set for early stopping

### 4.5 Visualization

After training, a graph is automatically generated:

- Blue line — error on the training set
- Orange line — error on the validation set

The graph helps:
- Visually assess overfitting (when validation error starts to increase)
- Choose the optimal number of trees
- Ensure the model is learning adequately

### 4.6 Saving and Loading Models

Trained models can be saved and loaded:

```python
# Saving
model.save("btc_vol_model")

# Loading
model.load("btc_vol_model")
```

When saving, the following are preserved:
- Models
- Feature normalization parameters

---
