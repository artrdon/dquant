## 6. Advanced Usage

### 6.1 Customizing Feature Engineering

#### 6.1.1 Passing a Custom Function

You can pass your own function for feature creation:

```python
def my_features(df):
    df = df.copy()
    
    # My unique features
    df['my_ratio'] = (df['high'] - df['low']) / df['close']
    df['my_momentum'] = df['close'].pct_change(5)
    df['my_volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    return df.dropna()

model.fit(
    df, 
    input_bars=70, 
    horizon=20, 
    trees_count=200, 
    show_results=True,
    feature_func=my_features
)
```
