## 1. Introduction

### 1.1 What is DQuant?

**DQuant** is an open-source Python library for automated volatility forecasting of financial time series using machine learning methods. The library handles all stages of model building: from raw price data to ready-made forecasts, hiding technical details (feature engineering, data splitting, visualization), while still allowing flexible customization for experienced users.

Key idea: **a trader doesn't need to know machine learning to use AI for volatility forecasting.**

### 1.2 Key Features

- **Automated feature engineering** — creates dozens of features from raw price data.
- **Automated target variable construction** — calculates realized volatility over a given horizon without look-ahead bias.
- **Intelligent data splitting** — preserves temporal structure when splitting into train/validation.
- **Gradient Boosting/XGBoost/LightGBM** with early stopping.
- **Training visualization** — error plot on training and validation sets.
- **Save and load models** — train once, use forever.
- **Flexible customization** — your own features, model parameters.
- **Integration with any data source** — Yahoo Finance, MetaTrader 5, and others.

### 1.3 Who is this library for

| Audience | Why they need DQuant                                   |
|:---|:-------------------------------------------------------|
| **Traders (algorithmic)** | For model calibration, risk management, setting stops  |
| **Traders (discretionary)** | For assessing market regime, determining position size |
| **Quantitative analysts** | For rapid volatility model prototyping                 |
| **Trading system developers** | For embedding forecasts into their pipelines           |
| **Students and researchers** | As a ready-made benchmark and learning example         |

### 1.4 Philosophy

DQuant is built on three principles:

1. **Simplicity for beginners** — you don't need to be a data scientist to use the library.
2. **Power for professionals** — flexible customization for those who want to dig deeper.
3. **Transparency** — you always see what's happening (graphs, metrics) and can interpret the results.
