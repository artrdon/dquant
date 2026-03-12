<div align="center">
  <h1>dquant</h1>
  <p><strong>Автоматическое прогнозирование волатильности для трейдеров и аналитиков</strong></p>
  <br>
  <img src="logo.png" alt="dquant demo" width="200">
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
from dquant.visual import Visualization


# 1. Загружаем данные
df = yf.download("BTC-USD", start="2020-01-01", interval='1d')
df = pd.DataFrame({
    'open': df[('Open', 'BTC-USD')].values,
    'high': df[('High', 'BTC-USD')].values,
    'low': df[('Low', 'BTC-USD')].values,
    'close': df[('Close', 'BTC-USD')].values,
    'volume': df[('Volume', 'BTC-USD')].values
}, index=df.index)

# 2. Создаем модель
model = VolClustXGB({}, default=True, early_stopping=True)

# 3. Обучаем модель
model.fit(df, input_bars=70, horizon=20, trees_count=200, show_results=True)


# 4. Делаем прогноз
rez = model.forecast(df.iloc[-70:].copy(), show=True)

```

### Результат выполнения

```
[0.0016554 0.0018979 0.0015921 0.0014239 0.0013767 0.0011586 0.0013139
 0.0009813 0.0007931 0.0012909 0.0013664 0.0016466 0.0014836 0.0011577
 0.0008737 0.0007213 0.0008084 0.0012699 0.0015358 0.0014748]
```
<img src="readmeforecast.png">

Красным показана волатильность за предыдущие свечи, а зеленым - будущая волатильность.

---

## Документация

| Ресурс | Описание |
|:-------|:---------|
| [**Полная документация**](https://github.com/artrdon/dquant/blob/main/docs.md) | Все классы, методы, параметры |



---

## Примеры использования

### С Yahoo Finance

```python
import pandas as pd
import yfinance as yf
from dquant.models import VolClustXGB 
from dquant.visual import Visualization


# 1. Загружаем данные
df = yf.download("BTC-USD", start="2020-01-01", interval='1d')
df = pd.DataFrame({
    'open': df[('Open', 'BTC-USD')].values,
    'high': df[('High', 'BTC-USD')].values,
    'low': df[('Low', 'BTC-USD')].values,
    'close': df[('Close', 'BTC-USD')].values,
    'volume': df[('Volume', 'BTC-USD')].values
}, index=df.index)

# 2. Создаем модель
model = VolClustXGB({}, default=True, early_stopping=True)

# 3. Обучаем модель
model.fit(df, input_bars=70, horizon=20, trees_count=200, show_results=True)

# 4. Делаем прогноз
rez = model.forecast(df.iloc[-70:].copy(), show=True)
```

### С MetaTrader 5

```python
import pandas as pd
import MetaTrader5 as mt5
from dquant.models import VolClustXGB 
from dquant.visual import Visualization


symbol = "EURUSD"          # какой символ смотреть
timeframe = mt5.TIMEFRAME_H1   # M1, M5, M15, H1, D1 и т.д.
days_back = 1000             # сколько дней истории загрузить

# Подключаемся к MT5
if not mt5.initialize():
    print("Не удалось подключиться к MetaTrader5")
    quit()

# Проверяем, что символ доступен
if not mt5.symbol_select(symbol, True):
    print(f"Символ {symbol} не найден или не включён")
    mt5.shutdown()
    quit()

# Вычисляем даты
to_date = dt.datetime.now() + dt.timedelta(hours=3)
from_date = to_date - dt.timedelta(days=days_back)

# Загружаем бары
rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)

mt5.shutdown()  # больше не нужен терминал

if rates is None or len(rates) == 0:
    print("Нет данных!")
    quit()

# Превращаем в DataFrame
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

df.rename(columns={
    'tick_volume': 'volume'
}, inplace=True)

# Создаем модель
model = VolClustXGB({}, default=True, early_stopping=True)

# Обучаем модель
model.fit(df, input_bars=70, horizon=20, trees_count=200, show_results=True)

# Делаем прогноз
rez = model.forecast(df.iloc[-70:].copy(), show=True)
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

Проект распространяется под лицензией MIT. Подробнее в файле [LICENSE](https://github.com/artrdon/dquant/blob/main/LICENSE).

---

## Поддержка проекта

Если **dquant** помог вам в работе или учёбе:

- Поставьте звезду на GitHub ⭐ — это очень мотивирует!
- Расскажите о библиотеке коллегам
- [Напишите мне](https://t.me/Denchik_ai) о вашем опыте использования

---

## Контакты

**Автор:** Денис Макаров

- Telegram: [@Denchik_ai](https://t.me/Denchik_ai)
- GitHub: [@artrdon](https://github.com/artrdon)
- Сайт проекта: [dquant.space](https://dquant.space)

---

<div align="center">
  <sub>
    Сделано с ❤️ для трейдеров и аналитиков
    <br>
    Если проект полезен — поставьте звезду!
  </sub>
</div>

