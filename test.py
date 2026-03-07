from main import VolClustGB
import MetaTrader5 as mt5
import datetime as dt
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import pandas as pd



# ===================== НАСТРОЙКИ =====================
symbol = "EURUSD"          # какой символ смотреть
timeframe = mt5.TIMEFRAME_D1   # M1, M5, M15, H1, D1 и т.д.
days_back = 3000             # сколько дней истории загрузить
# ====================================================

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
df['time'] = pd.to_datetime(df['time'], unit='s')  # правильное время
df.set_index('time', inplace=True)

# Переименуем колонки под mplfinance
df.rename(columns={
    'tick_volume': 'volume'
}, inplace=True)

print(f"Загружено {len(df)} баров с {df.index[0]} по {df.index[-1]}")

model = VolClustGB()
model.fit(df, 20, 20, 5, True)

# Вывод результатов
#print(np.array(df[-30:]))
print(model.forecast(df.iloc[-20:].copy()))
#model.forecast(df.iloc[-20:].copy())
