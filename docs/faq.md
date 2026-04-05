## 8. Frequently Asked Questions (FAQ)

### 8.1 General Questions

**Q: Why should I forecast volatility?**

A: Volatility is used for:
- Risk management (stop-losses, position sizing)
- Strategy selection (some strategies work in high volatility, others in low volatility)
- Fair price assessment (strong movements relative to expected volatility may be signals)
- Hedging

**Q: How is DQuant different from other volatility forecasting libraries?**

A: DQuant offers a unique combination:
- Full automation — no need to think about feature engineering and data collection
- Simplicity for beginners — one line of code for training
- Flexibility for pros — customization of any stage
- Specialization specifically in volatility (unlike general ML libraries)

**Q: Can I use DQuant for cryptocurrencies?**

A: Yes, the library works with any time series, including cryptocurrencies. Data from Binance, Bybit, and other exchanges is suitable.

**Q: Do I need to know machine learning?**

A: No. The library is designed so that traders without ML experience can use it. If you can load data into a pandas DataFrame, you can use DQuant.

**Q: Can I use DQuant in live trading?**

A: Yes, the library is intended for real-world use. Models can be saved and loaded.

### 8.2 Data Questions

**Q: What is the minimum amount of data needed for training?**

A: At least 10,000 candles is recommended. More data allows the model to learn better.

**Q: Does the library support tick data?**

A: No, the library works with candle data (open, close, high, low, volume).

### 8.3 Model Questions
**Q: What if the model overfits?**

A: Use the training graph to determine the optimal number of trees. Increase `early_stopping_rounds`, add regularization via model parameters. Increase the amount of data. Reduce model complexity.

**Q: Can I use the model for different assets?**

A: It is recommended to train a separate model for each asset, as they have different characteristics.
