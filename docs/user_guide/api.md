## 5. User Guide

### 5.1 VolClustGB, VolClustXGB, VolClustLightGBM Classes


| Object | Description |
|:-------------------|:---|
| `__version__` | Library version |
| `VolClustGB` | Class for volatility forecasting using Gradient Boosting |
| `VolClustXGB` | Class for volatility forecasting using XGBoost |
| `VolClustLightGBM` | Class for volatility forecasting using LightGBM |
#### 5.1.1 Initialization Parameters

| Parameter | Type | Default | Description |
|:-----------------|:---|:-------------------------------------|:---|
| `sett` | dict | `Default model hyperparameters` | Model hyperparameters |
| `early_stopping` | bool | `True` | Enable early stopping |

**Example:**
```python
model = VolClustGB({}, early_stopping=True)
```

#### 5.1.2 fit Method

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
model.fit(
    df,
    feature_list=features,
    input_bars=70, 
    horizon=20, 
    trees_count=200, 
    show_results=True
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|:---------------|:-------------|:-------------|:--------------------------------------------------------------------|
| `data` | pd.DataFrame | required | Input data |
| `feature_list` | list | required | Features for training |
| `input_bars` | int | required | Number of candles used for forecasting |
| `horizon` | int | required | Forecast horizon (number of steps ahead) |
| `trees_count` | int | required | Maximum number of trees |
| `show_results` | bool | False | Whether to show the plot after training |
| `feature_func` | function | None | Use custom function for feature creation |
| `target_func` | function | None | Use custom function for target creation |

#### 5.1.3 forecast Method

```python
prediction = model.forecast(df)
```

**Parameters:**

| Parameter | Type | Default | Description |
|:---|:-------------|:-----------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|
| `latest_data` | pd.DataFrame | required | Data for forecast (must contain same columns as during training) |
| `feature_func` | function | None (required if custom function was used during training) | Use custom function for feature creation if a custom function was used during training |
| `show` | bool | False | Display forecast results |

Returns volatility forecast for the horizon specified during training.

#### 5.1.4 save Method

```python
model.save('name')
```

**Parameters:**

| Parameter | Type | Default | Description |
|:---|:---|:-------------|:-----------------------------------------------------|
| `name` | str | required | Name under which to save the directory and models |
| `type_to_save` | str | 'default' | Format to save models ('default', 'mql5') |

**Saving to mql5**

```python
model.save('name', type_to_save='mql5')
```
Done! Now you can use your trained models in Meta Trader 5.

#### 5.1.5 load Method

```python
model.load('name')
```

**Parameters:**

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `name` | str | required | Name of the directory with saved models |

#### 5.1.6 show_train_results Method

```python
model.show_train_results()
```

Shows the error plot after training.
