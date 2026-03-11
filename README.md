<div align="center">
  <h1>dquant</h1>
  <p><strong>Автоматическое прогнозирование волатильности для трейдеров и аналитиков</strong></p>
  <p>
    <a href="https://pypi.org/project/dquant/">
      <img src="https://img.shields.io/pypi/v/dquant.svg?color=blue" alt="PyPI version">
    </a>
    <a href="https://pypi.org/project/dquant/">
      <img src="https://img.shields.io/pypi/pyversions/dquant.svg?color=blue" alt="Python versions">
    </a>
    <a href="https://github.com/artrdon/dquant/blob/main/LICENSE">
      <img src="https://img.shields.io/github/license/artrdon/dquant.svg?color=blue" alt="License: MIT">
    </a>
    <a href="https://pepy.tech/project/dquant">
      <img src="https://pepy.tech/badge/dquant" alt="Downloads">
    </a>
    <a href="https://github.com/artrdon/dquant/stargazers">
      <img src="https://img.shields.io/github/stars/artrdon/dquant?style=social" alt="GitHub stars">
    </a>
  </p>
  <p>
    <a href="#-установка">Установка</a> •
    <a href="#-быстрый-старт">Быстрый старт</a> •
    <a href="#-документация">Документация</a> •
    <a href="#-примеры">Примеры</a> •
    <a href="#-вклад">Вклад</a>
  </p>
  <br>
  <img src="https://via.placeholder.com/800x400/0A1929/FFFFFF?text=dquant+volatility+forecast" alt="dquant demo" width="600">
  <br>
  <p><i>Прогноз волатильности с помощью dquant</i></p>
</div>

---

## О проекте

**dquant** — это Python-библиотека с открытым исходным кодом для автоматического прогнозирования волатильности финансовых временных рядов. Она берёт на себя все этапы построения модели: от сырых цен до готового прогноза.

### Ключевая идея
> **Трейдеру не нужно знать машинное обучение, чтобы использовать ИИ для прогнозирования волатильности.**

### Возможности

| | |
|---|---|
| **Автоматический feature engineering** | Из сырых цен (open, high, low, close, volume) создаются десятки признаков |
| **Целевая переменная без look-ahead bias** | Корректный расчёт реализованной волатильности |
| **3 модели на выбор** | Градиентный бустинг, XGBoost, LightGBM с ранней остановкой |
| **Визуализация обучения** | График ошибок на train/validation для контроля переобучения |
| **Сохранение и загрузка** | Обучил один раз — используй всегда |
| **Гибкая кастомизация** | Свои признаки, параметры модели, источники данных |
| **Интеграция с любыми данными** | Yahoo Finance, MetaTrader 5 |

### Для кого это

- **Алгоритмические трейдеры** — для калибровки моделей и управления рисками
- **Дискреционные трейдеры** — для оценки рыночного режима и размера позиции
- **Количественные аналитики** — для быстрого прототипирования
- **Разработчики** — для встраивания в торговые системы
- **Студенты** — как готовый бенчмарк и учебный пример

---

## Установка

### Требования
- Python 3.7 или выше
- pip


```bash
pip install dquant
```


### Проверка установки

```python
import dquant
print(dquant.__version__)  # Должно вывести версию
```

---

## Быстрый старт

### Минимальный рабочий пример с Bitcoin

```python
import pandas as pd
import yfinance as yf
from dquant.models import VolClustXGB

# 1. Загружаем данные (Yahoo Finance)
df = yf.download("BTC-USD", start="2020-01-01", end="2024-01-01")
df.columns = ['open', 'high', 'low', 'close', 'volume']  # Переименовываем колонки

# 2. Создаём модель
model = VolClustXGB({}, default=True, early_stopping=True)

# 3. Обучаем
model.fit(
    df, 
    input_bars=70,      # используем 70 свечей для прогноза
    horizon=20,         # прогнозируем на 20 дней вперёд
    trees_count=200,    # максимум 200 деревьев
    show_results=True   # показать график обучения
)

# 4. Прогнозируем
last_bars = df.iloc[-70:].copy()
prediction = model.forecast(last_bars)

print(f"\n📈 Прогноз волатильности на 20 дней: {prediction:.2%}")
print(f"    (означает ожидаемое движение {prediction:.2%} от текущей цены)")
```

### Результат выполнения

```
Обучение модели: 100%|██████████| 200/200 [00:15<00:00]
Лучшая модель на итерации 156

📈 Прогноз волатильности на 20 дней: 3.45%
    (означает ожидаемое движение 3.45% от текущей цены)
```

---

## Документация

| Ресурс | Описание |
|:-------|:---------|
| [**Полная документация**](https://dquant.space/docs) | Все классы, методы, параметры |



---

## Примеры использования

### С Yahoo Finance

```python
import yfinance as yf
from dquant.models import VolClustXGB

# Любой тикер: BTC-USD, AAPL, EURUSD=X
df = yf.download("AAPL", start="2015-01-01")
df.columns = ['open', 'high', 'low', 'close', 'volume']

model = VolClustXGB({}, default=True)
model.fit(df, input_bars=70, horizon=20)
pred = model.forecast(df.iloc[-70:])
print(f"Прогноз для AAPL: {pred:.2%}")
```

### С MetaTrader 5

```python
import MetaTrader5 as mt5
import pandas as pd
import datetime as dt
from dquant.models import VolClustXGB

# Подключение к MT5
if not mt5.initialize():
    print("Ошибка подключения к MT5")
    quit()

# Загрузка данных
rates = mt5.copy_rates_range("GOLD", mt5.TIMEFRAME_D1, 
                            dt.datetime.now() - dt.timedelta(days=1000),
                            dt.datetime.now())
mt5.shutdown()

# Подготовка данных
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df.rename(columns={'tick_volume': 'volume'}, inplace=True)

# Обучение и прогноз
model = VolClustXGB({}, default=True)
model.fit(df, input_bars=70, horizon=20)
print(f"Прогноз для GOLD: {model.forecast(df.iloc[-70:])}")
```
---

## Как внести вклад

Мы приветствуем любой вклад в проект! Вот несколько способов помочь:

### Сообщить об ошибке
Нашли баг? [Создайте Issue](https://github.com/artrdon/dquant/issues) с подробным описанием:
- Что делали
- Что ожидали
- Что произошло на самом деле
- Код для воспроизведения (если возможно)

### Предложить идею
Есть идея по улучшению? [Напишите в Telegram](https://t.me/Denchik_ai) или создайте Issue с меткой `enhancement`.


---

## Лицензия

Проект распространяется под лицензией MIT. Подробнее в файле [LICENSE](LICENSE).

---

## Поддержка проекта

Если **dquant** помог вам в работе или учёбе:

- Поставьте звезду на GitHub ⭐ — это очень мотивирует!
- Расскажите о библиотеке коллегам
- [Напишите мне](https://t.me/Denchik_ai) о вашем опыте использования

---

## Контакты

**Автор:** Денис Макаров

- 📱 Telegram: [@Denchik_ai](https://t.me/Denchik_ai)
- 📧 Email: *ваш email*
- 💻 GitHub: [@artrdon](https://github.com/artrdon)
- 🌐 Сайт проекта: [dquant.space](https://dquant.space)

---

<div align="center">
  <sub>
    Сделано с ❤️ для трейдеров и аналитиков
    <br>
    Если проект полезен — поставьте звезду!
  </sub>
</div>

