import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import datetime as dt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DirectForecaster:
    def __init__(self, base_model, n_steps):
        self.base_model = base_model
        self.n_steps = n_steps
        self.models = []
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X, y, horizons):
        """
        Обучаем отдельную модель для каждого горизонта

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Признаки
        y : array-like, shape (n_samples, n_targets) или (n_samples,)
            Целевая переменная. Если 2D, то каждая колонка - свой горизонт
        horizons : int или list
            Количество горизонтов или список горизонтов
        """
        # Масштабируем X один раз для всех моделей
        X_scaled = self.scaler.fit_transform(X)

        # Определяем горизонты
        if isinstance(horizons, int):
            horizons_list = list(range(horizons))
        else:
            horizons_list = horizons

        # Если y - 2D (уже с горизонтами)
        if len(y.shape) == 2 and y.shape[1] > 1:
            print(f"y имеет форму {y.shape}, используем каждую колонку как отдельный горизонт")
            for h_idx, h in enumerate(horizons_list):
                if h_idx >= y.shape[1]:
                    print(f"Предупреждение: горизонт {h} выходит за пределы y, пропускаем")
                    continue

                # Берем конкретную колонку для этого горизонта
                y_h = y.iloc[:, h_idx] if hasattr(y, 'iloc') else y[:, h_idx]

                # Убираем NaN
                valid_mask = ~pd.isna(y_h) if hasattr(y_h, 'isna') else ~np.isnan(y_h)
                X_h = X_scaled[valid_mask]
                y_h_clean = y_h[valid_mask]

                #print(f"Горизонт {h}: X shape {X_h.shape}, y shape {y_h_clean.shape}")
                percent = int(100.0 / len(y[0]) * (h+1) * 100)
                if percent < 10:
                    print(f'{percent}%', " | ", '#'*(h+1))
                else:
                    print(f'{percent}%', "| ", '#' * (h + 1))

                # Обучаем модель
                model = self.base_model.__class__(**self.base_model.get_params())
                model.fit(X_h, y_h_clean)
                self.models.append(model)

        # Если y - 1D (нужно сдвигать)
        else:
            print(f"y 1D, создаем сдвиги для каждого горизонта")
            y_series = y if isinstance(y, pd.Series) else pd.Series(y.flatten())

            for h in horizons_list:
                # Сдвигаем целевую переменную
                y_h = y_series.shift(-h)

                # Убираем NaN
                y_h_clean = y_h.dropna()
                X_h = X_scaled[:len(y_h_clean)]

                print(f"Горизонт t+{h}: X shape {X_h.shape}, y shape {y_h_clean.shape}")

                # Обучаем модель
                model = self.base_model.__class__(**self.base_model.get_params())
                model.fit(X_h, y_h_clean)
                self.models.append(model)

        self.is_fitted = True
        print(f"Обучено {len(self.models)} моделей")

    def predict(self, X):
        """
        Предсказание для всех горизонтов

        X : array-like, shape (n_samples, n_features)
            Признаки для предсказания
        """
        if not self.is_fitted:
            raise ValueError("Модель еще не обучена!")

        # Масштабируем X
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Предсказываем каждой моделью
        predictions = []
        for model in self.models:
            pred = model.predict(X_scaled)
            # Если предсказание для нескольких образцов, берем первый
            if len(pred.shape) > 0 and pred.shape[0] > 1:
                predictions.append(pred)
            else:
                predictions.append(pred[0] if len(pred.shape) > 0 else pred)

        return np.array(predictions)

    def predict_proba(self, X):
        """Для совместимости с sklearn API"""
        return self.predict(X)


if __name__ == '__main__':

    # ===================== НАСТРОЙКИ =====================
    symbol = "EURUSD"  # какой символ смотреть
    timeframe = mt5.TIMEFRAME_D1  # M1, M5, M15, H1, D1 и т.д.
    days_back = 3000  # сколько дней истории загрузить
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

    model = GradientBoostingRegressor(
        loss='squared_error',
        learning_rate=0.01,
        n_estimators=100,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42,
        warm_start=True
    )
    p = DirectForecaster(model, 20)
    x_data = []
    y_data = []

    for i in range(len(arr)):
        if i <= 20:
            continue
        elif i >= (len(arr) - 20):
            break
        else:
            #print(arr[i - 20:i].values)
            x_data.append(arr[i - 20:i].values)
            y_data.append(arr[i: i+20].values)

    x_data = pd.DataFrame(x_data)
    y_data = pd.DataFrame(y_data)
    print(len(x_data))
    #print(x_data[0])
    p.fit(x_data, y_data, 20)

    print(p.predict(arr[len(arr)-1 - 20:(len(arr)-1)].values))