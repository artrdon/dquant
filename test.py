import os
from main import VolClustGB, Visualiser
import MetaTrader5 as mt5
import datetime as dt
import numpy as np
from arch import arch_model
import pandas as pd






if __name__ == '__main__':

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



    #df = df.iloc[-20:]
    # Параметры для нижнего расположения
    #show_vol(df)

    print(f"Загружено {len(df)} баров с {df.index[0]} по {df.index[-1]}")

    settings = {
        'loss': 'squared_error',
        'learning_rate': 0.1,
        'n_estimators': 1,
        'max_depth': 4,
        'min_samples_split': 5,
        'min_samples_leaf': 5,
        'subsample': 0.8,
        'random_state': 42,
        'warm_start': True
    }
    model = VolClustGB(settings)
    visualizer = Visualiser()

    while True:
        match input(">> "):
            case "clear":
                os.system('cls')
            case "train":
                model.fit(df, 70, 20, 150, True)
            case "show":
                rez = model.forecast(df.iloc[-70:].copy())
                print(rez)
                dates = pd.date_range(start='2024-01-01', periods=len(rez), freq='D')
                rez = pd.DataFrame(rez, index=dates, columns=['value'])
                visualizer.show_vol(rez)
            case "save":
                model.save("mod")
            case "load":
                model.load("mod")
            case "exit":
                break

    # Вывод результатов
    #print(np.array(df[-30:]))

    #model.forecast(df.iloc[-20:].copy())
