from main import VolClust
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
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'tick_volume': 'Volume'
}, inplace=True)

print(f"Загружено {len(df)} баров с {df.index[0]} по {df.index[-1]}")
# Расчет логарифмических доходностей
df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
arr = df['returns']
arr = arr[-100:]
mu = arr.dropna().mean()
print("trend: ", mu)
sig = arr.std()
print("sigma: ", sig)
mu = mu + 0.5 * sig**2
print("trend monte carlo: ", mu)
print("price: ", df['Close'].iloc[-1])

data = (df.dropna())

# Обучение модели GARCH(1,1)
returns = data['returns'].values

# Создание модели GARCH(1,1) с нормальным распределением
model = arch_model(returns, vol='GARCH', p=1, q=1, o=1, dist='normal')
model_fit = model.fit(disp='off')  # disp='off' отключает вывод подробной информации

# Вывод результатов
print(model_fit.summary())

# Визуализация условной волатильности
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
ax[0].plot(data.index, returns)
ax[0].set_title('Доходности')
ax[0].set_ylabel('Доходность (%)')
ax[0].grid(True)

ax[1].plot(data.index, model_fit.conditional_volatility)
ax[1].set_title('Условная волатильность (GARCH(1,1))')
ax[1].set_ylabel('Волатильность (%)')
ax[1].set_xlabel('Дата')
ax[1].grid(True)

plt.tight_layout()
plt.show()

# Расчет долгосрочной (безусловной) волатильности
omega = model_fit.params['omega']
alpha = model_fit.params['alpha[1]']
beta = model_fit.params['beta[1]']
unconditional_variance = omega / (1 - alpha - beta)
unconditional_volatility = np.sqrt(unconditional_variance)
print(f"Долгосрочная волатильность: {unconditional_volatility:.4f}%")
print(f"Персистентность волатильности (alpha + beta): {alpha + beta:.4f}")

# Расчет времени полураспада шока волатильности
half_life = np.log(0.5) / np.log(alpha + beta)
print(f"Время полураспада шока волатильности: {half_life:.1f} дней")


# Получение стандартизированных остатков
resid = model_fit.resid
std_resid = resid / model_fit.conditional_volatility

# Диагностические графики
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# 1. Стандартизированные остатки
ax[0, 0].plot(data.index, std_resid)
ax[0, 0].set_title('Стандартизированные остатки')
ax[0, 0].grid(True)

# 2. QQ-график стандартизированных остатков
import scipy.stats as stats
stats.probplot(std_resid, dist='norm', plot=ax[0, 1])
ax[0, 1].set_title('QQ-график стандартизированных остатков')
ax[0, 1].grid(True)

# 3. ACF квадратов стандартизированных остатков
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(std_resid**2, lags=20, ax=ax[1, 0])
ax[1, 0].set_title('ACF квадратов стандартизированных остатков')

# 4. Гистограмма стандартизированных остатков
ax[1, 1].hist(std_resid, bins=50, alpha=0.7, density=True)
x = np.linspace(-5, 5, 1000)
ax[1, 1].plot(x, stats.norm.pdf(x, 0, 1), 'r', lw=2)
ax[1, 1].set_title('Гистограмма стандартизированных остатков')
ax[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Тест Льюнга-Бокса на автокорреляцию в квадратах остатков
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test_std = acorr_ljungbox(std_resid**2, lags=[10], return_df=True)
print(f"Тест Льюнга-Бокса на автокорреляцию в квадратах стандартизированных остатков:")
print(lb_test_std)

# Тест Харке-Бера на нормальность остатков
from scipy.stats import jarque_bera
jb_test = jarque_bera(std_resid)
print(f"Тест Харке-Бера на нормальность стандартизированных остатков:")
print(f"Статистика: {jb_test[0]:.4f}, p-значение: {jb_test[1]:.4f}")


# Выбор лучшей модели на основе информационных критериев
best_model = model_fit  # Предположим, что GJR-GARCH оказалась лучшей

# Прогнозирование на 30 дней вперед
forecast_horizon = 30
forecasts = best_model.forecast(
    horizon=forecast_horizon,
    method='simulation',
    simulations=10000)

# Извлечение прогнозов волатильности
forecast_variance = forecasts.variance.iloc[-1].values
forecast_volatility = np.sqrt(forecast_variance)

# Визуализация прогноза
plt.figure(figsize=(12, 6))
plt.plot(data.index[-100:], best_model.conditional_volatility[-100:], label='Историческая волатильность')
last_date = data.index[-1]
forecast_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1)[1:]
plt.plot(forecast_dates, forecast_volatility, 'r--', label='Прогноз волатильности')
plt.title('Прогноз условной волатильности')
plt.xlabel('Дата')
plt.ylabel('Условная волатильность (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Вывод прогнозных значений
print("\nПрогноз волатильности на ближайшие дни:")
for i, (date, vol) in enumerate(zip(forecast_dates[:24], forecast_volatility[:24])):
    print(f"{date.date()}: {vol:.4f}%")

# Долгосрочный прогноз волатильности
long_term_volatility = forecast_volatility[-1]
print(f"\nДолгосрочный прогноз волатильности (через {forecast_horizon} дней): {long_term_volatility:.4f}%")

