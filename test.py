import os
from main import VolClustGB
import MetaTrader5 as mt5
import datetime as dt
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd


def show_vol(df):
    fig, ax = plt.subplots(figsize=(15, 6))
    bottom_y = 0  # нижняя граница
    candle_height = 0.8  # высота свечи
    max_bar_height = 0
    for i, (idx, row) in enumerate(df.iterrows()):
        x_pos = i  # позиция по x

        # Определяем цвет свечи
        color = 'green'
        ax.plot([x_pos, x_pos], [bottom_y, bottom_y],
                color='black', linewidth=1)

        bar_height = row['value']
        if bar_height > max_bar_height:
            max_bar_height = bar_height
        # Свеча как прямоугольник
        rect = patches.Rectangle((x_pos - 0.3, 0), 0.6, bar_height,
                                 linewidth=1, edgecolor=color, facecolor=color, alpha=0.7)
        ax.add_patch(rect)

    # Настройка осей
    ax.set_xlim(-1, len(df))
    ax.set_ylim(0, max_bar_height * 1.3)
    ax.set_xlabel('Время')
    ax.set_ylabel('')
    ax.set_title('Свечи внизу графика (горизонтальное представление)')

    # Поворачиваем метки дат
    plt.xticks(range(len(df)), [d.strftime('%m-%d') for d in df.index], rotation=45)
    plt.tight_layout()
    plt.show()



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

    while True:
        match input(">> "):
            case "clear":
                os.system('cls')
            case "train":
                model.fit(df, 70, 20, 400, True)
            case "show":
                rez = model.forecast(df.iloc[-20:].copy())
                print(rez)
                dates = pd.date_range(start='2024-01-01', periods=len(rez), freq='D')
                rez = pd.DataFrame(rez, index=dates, columns=['value'])
                show_vol(rez)
            case "save":
                model.save("mod")
            case "load":
                model.load("mod")
            case "exit":
                break

    # Вывод результатов
    #print(np.array(df[-30:]))

    #model.forecast(df.iloc[-20:].copy())
