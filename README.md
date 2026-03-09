# dquant <code>v0.1.0</code>

**ИИ для прогнозирования волатильности в одну строку** – никакого ручного фичеинжиниринга, разделения выборок и борьбы с переобучением. Просто дайте данные и получите прогноз.

<div align="center">
  <p>
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/dquant">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/dquant">
    <img alt="License" src="https://img.shields.io/pypi/l/dquant">
    <img alt="Downloads" src="https://pepy.tech/badge/dquant">
  </p>
</div>

bash
pip install dquant
Markup



---

📋 Содержание

· Зачем это нужно?
· Возможности
· Быстрый старт
· Как это работает под капотом
· Визуализация и диагностика
· Сохранение и загрузка моделей
· Кастомизация для профи
· Примеры
· Документация
· Поддержка и вклад
· Лицензия

---

🧐 Зачем это нужно?

Волатильность — ключевая метрика в финансах:

· размер позиции (чем выше волатильность, тем меньше лот);
· уровни стоп‑лоссов и тейк‑профитов;
· оценка рыночного режима (тренд/флэт);
· хеджирование рисков;
· ценообразование опционов.

Традиционные методы (историческая волатильность, EWMA, GARCH) часто грубы и не учитывают нелинейные зависимости. Машинное обучение способно дать точный прогноз, но его внедрение требует серьёзной подготовки данных, знания статистики и программирования. dquant решает эту проблему раз и навсегда.

---

✨ Возможности

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; margin: 20px 0;">

  <div style="border: 1px solid #e1e4e8; border-radius: 10px; padding: 16px; background: #f6f8fa;">
    <strong>⚙️ Автоматический фичеинжиниринг</strong><br>
    Создаёт десятки признаков из сырых цен: лаги, скользящие статистики, ATR, отношения цен, день недели и др.
  </div>

  <div style="border: 1px solid #e1e4e8; border-radius: 10px; padding: 16px; background: #f6f8fa;">
    <strong>🎯 Целевая переменная</strong><br>
    Автоматически строится как реализованная волатильность на заданном горизонте (без look‑ahead bias).
  </div>

  <div style="border: 1px solid #e1e4e8; border-radius: 10px; padding: 16px; background: #f6f8fa;">
    <strong>🧪 Разделение train/validation</strong><br>
    С учётом временной структуры — никакого случайного перемешивания, данные не заглядывают в будущее.
  </div>

  <div style="border: 1px solid #e1e4e8; border-radius: 10px; padding: 16px; background: #f6f8fa;">
    <strong>🌲 Градиентный бустинг</strong><br>
    XGBoost / LightGBM с ранней остановкой и визуализацией ошибок.
  </div>

  <div style="border: 1px solid #e1e4e8; border-radius: 10px; padding: 16px; background: #f6f8fa;">
    <strong>💾 Сохранение/загрузка</strong><br>
    Обучил один раз — используй всегда. Модель сохраняется вместе с нормализацией.
  </div>

  <div style="border: 1px solid #e1e4e8; border-radius: 10px; padding: 16px; background: #f6f8fa;">
    <strong>🔧 Полная кастомизация</strong><br>
    Свои признаки, параметры модели, наследование классов — для тех, кто хочет копать глубже.
  </div>

</div>

---

⚡ Быстрый старт

python
import yfinance as yf
from dquant import VolatilityGB

# 1. Загружаем данные
df = yf.download("BTC-USD", start="2020-01-01")

# 2. Создаём модель
model = VolatilityGB()

# 3. Обучаем (одна строка — вся магия)
model.fit(df, horizon=20, max_trees=200)

# 4. Получаем прогноз на последние 50 свечей
prediction = model.predict(df.iloc[-50:])
print(f"Прогноз волатильности на следующие 20 дней: {prediction:.2%}")
`

5 строк. 0 знаний ML. Работающий ИИ.

---

🔧 Как это работает под капотом

Метод .fit() скрывает сложный, но полностью прозрачный пайплайн:
1. Валидация входных данных – проверка наличия колонок open, high, low, close, volume.
2. Генерация признаков – лаги, скользящие средние и стандартные отклонения, ATR, волатильность Паркинсона, отношения цен, день недели и др.
3. Построение целевой переменной – логарифмические доходности → реализованная волатильность на горизонте horizon (без использования будущих значений).
4. Разделение на train/validation – последние 20% данных откладываются для валидации, временной порядок сохраняется.
5. Обучение градиентного бустинга – ранняя остановка, максимальное число деревьев max_trees.
6. Визуализация – график ошибок на train и validation.

---

📊 Визуализация и диагностика

После обучения .fit() автоматически показывает график:

Python


model.fit(df, horizon=20, max_trees=200, plot=True)   # plot=True по умолчанию
<div align="center">
  <img src="https://via.placeholder.com/800x400?text=Train/Validation+error+curves" alt="График ошибок" style="max-width:100%; border-radius:8px;">
</div>

· Синяя линия – ошибка на train.
· Оранжевая линия – ошибка на validation.
· Вертикальная черта – оптимальное число деревьев (минимум валидационной ошибки).

Это помогает вовремя остановиться и избежать переобучения.

---

💾 Сохранение и загрузка моделей

Python


model.save("btc_vol_2024.model")

# ... через месяц
model.load("btc_vol_2024.model")
new_data = yf.download("BTC-USD", start="2025-01-01")
pred = model.predict(new_data.iloc[-50:])
Модель сохраняется вместе со всей внутренней нормализацией – можно использовать без повторного обучения.

---

🔬 Кастомизация для профи

1. Свои признаки

Python


def my_features(df):
    df = df.copy()
    df['my_signal'] = (df['close'] - df['open']) / df['close']
    return df

model.fit(df, horizon=20, feature_engineering=my_features)
2. Параметры модели

Python


model = VolatilityGB(
    base_model='lightgbm',                     # или 'xgboost'
    default_params={
        'num_leaves': 31,
        'reg_alpha': 0.1,
        'min_child_samples': 20
    }
)
3. Наследование класса

Python


class MyModel(VolatilityGB):
    def create_features(self, df):
        # полностью своя логика
        return features

    def create_target(self, df, horizon):
        # своя целевая переменная
        return target

model = MyModel()
model.fit(df, horizon=20)
---

📈 Примеры

С акциями Apple из CSV

Python


import pandas as pd

df = pd.read_csv('AAPL.csv', parse_dates=['Date'], index_col='Date')
df.columns = ['open', 'high', 'low', 'close', 'volume']

model = VolatilityGB()
model.fit(df, horizon=10, max_trees=150)
pred = model.predict(df.iloc[-30:])
print(f'10‑дневная прогнозная волатильность: {pred:.2%}')
С MetaTrader5

Python


import MetaTrader5 as mt5
import pandas as pd

mt5.initialize()
rates = mt5.copy_rates_from_pos('EURUSD', mt5.TIMEFRAME_H1, 0, 5000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'tick_volume': 'volume'}, inplace=True)

model = VolatilityGB()
model.fit(df[['open', 'high', 'low', 'close', 'volume']], horizon=24)
Оценка качества (backtesting)

Python


from dquant.metrics import realized_volatility, mae, rmse

true_vol = realized_volatility(returns)
pred_vol = model.predict(df)   # для соответствующих дат
print(f'MAE: {mae(true_vol, pred_vol):.4f}')
print(f'RMSE: {rmse(true_vol, pred_vol):.4f}')
---

📚 Документация

Полная документация доступна на Read the Docs (появится после публикации).
Прямо в Python можно вызвать:

Python


from dquant import help
help()   # краткая справка и ссылки
---

🤝 Поддержка и вклад

Библиотека открыта для сотрудничества!

· 🐛 Сообщить о баге / предложить идею – GitHub Issues
· 💬 Задать вопрос – Telegram‑чат пользователей
· 🔧 Pull Request – форкните репозиторий, создайте ветку и отправьте PR.

Особенно приветствуются:

· тестирование на реальных данных;
· улучшение фичеинжиниринга;
· дополнение документации;
· примеры использования.

---
📄 Лицензия

MIT – используйте свободно, в том числе в коммерческих продуктах.

---

<div align="center">
  <sub>⭐ Если библиотека оказалась полезной, поставьте звезду на <a href="https://github.com/yourname/dquant">GitHub</a> – это лучшая благодарность.</sub>
</div>

