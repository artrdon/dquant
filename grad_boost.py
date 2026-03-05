import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import datetime as dt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DirectForecaster:
    def __init__(self, base_model):
        self.base_model = base_model
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
                percent = int(100.0 / horizons * (h+1))
                if percent < 10:
                    print(f'Model is trained for {percent}%', "  | ", '|'+'#'*(h+1)+' '*(horizons-(h+1)) + '|')
                elif percent >= 10 and percent < 100:
                    print(f'Model is trained for {percent}%', " | ", '|'+'#' * (h + 1)+' '*(horizons-(h+1)) + '|')
                else:
                    print(f'Model is trained for {percent}%', "| ", '|' + '#' * (h + 1) + ' ' * (horizons - (h + 1)) + '|')

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
        X = prepare_single_window_features(X)
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


def prepare_volatility_features(
        df: pd.DataFrame,
        window_in: int = 20,
        window_out: int = 20,
        include_volume: bool = True,
        add_rolling: bool = True,
        epsilon: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Преобразует OHLCV данные в фичи и таргеты для предсказания волатильности.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с колонками ['open', 'high', 'low', 'close', 'volume']
    window_in : int
        Размер входного окна (сколько прошлых свечей используем)
    window_out : int
        Размер выходного окна (сколько будущих TR предсказываем)
    include_volume : bool
        Использовать ли объём в фичах
    add_rolling : bool
        Добавлять ли скользящие статистики (ATR, и т.д.)
    epsilon : float
        Для защиты от деления на ноль

    Returns:
    --------
    X : np.ndarray
        Матрица фичей (samples, features)
    y : np.ndarray
        Матрица таргетов (samples, window_out) - будущие значения TR
    feature_names : List[str]
        Названия фичей (для понимания)
    """

    # Копируем, чтобы не портить оригинал
    data = df.copy()

    # Проверяем наличие необходимых колонок
    required = ['open', 'high', 'low', 'close']
    if include_volume:
        required.append('volume')

    for col in required:
        if col not in data.columns:
            raise ValueError(f"Колонка '{col}' не найдена в DataFrame")

    # === 1. Базовые фичи (всегда) ===

    # True Range
    data['high_low'] = data['high'] - data['low']
    data['high_close_prev'] = abs(data['high'] - data['close'].shift(1))
    data['low_close_prev'] = abs(data['low'] - data['close'].shift(1))
    data['TR'] = data[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)

    # Простой диапазон (как запасной вариант)
    data['range'] = data['high'] - data['low']

    # Доходности
    data['return'] = np.log(data['close'] / data['close'].shift(1))
    data['abs_return'] = abs(data['return'])
    data['return_squared'] = data['return'] ** 2

    # Структура свечи
    data['body'] = abs(data['close'] - data['open'])
    data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
    data['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']
    data['body_to_range'] = data['body'] / (data['range'] + epsilon)
    data['shadow_to_body'] = (data['upper_shadow'] + data['lower_shadow']) / (data['body'] + epsilon)

    # Позиция закрытия
    data['close_position'] = (data['close'] - data['low']) / (data['range'] + epsilon)

    # Гэпы
    data['gap'] = data['open'] - data['close'].shift(1)
    data['abs_gap'] = abs(data['gap'])

    # === 2. Объёмные фичи (если нужно) ===
    if include_volume:
        data['log_volume'] = np.log(data['volume'] + 1)
        data['volume_TR'] = data['volume'] * data['TR']
        data['volume_range'] = data['volume'] * data['range']

    # === 3. Скользящие статистики (если нужно) ===
    if add_rolling:
        # ATR для разных периодов
        for period in [5, 10, 20]:
            data[f'ATR_{period}'] = data['TR'].rolling(window=period, min_periods=1).mean()
            data[f'TR_vs_ATR_{period}'] = data['TR'] / (data[f'ATR_{period}'] + epsilon)

        # Стандартное отклонение доходностей
        for period in [5, 10, 20]:
            data[f'return_std_{period}'] = data['return'].rolling(window=period, min_periods=1).std()
            data[f'abs_return_sma_{period}'] = data['abs_return'].rolling(window=period, min_periods=1).mean()

        # Скользящие средние объёма
        if include_volume:
            for period in [5, 10, 20]:
                data[f'volume_sma_{period}'] = data['volume'].rolling(window=period, min_periods=1).mean()
                data[f'volume_ratio_{period}'] = data['volume'] / (data[f'volume_sma_{period}'] + epsilon)

    # Макро-состояние
    for period in [10, 20]:
        data[f'dist_from_high_{period}'] = (data['high'].rolling(window=period, min_periods=1).max() - data['close']) / (
                    data['range'] + epsilon)
        data[f'dist_from_low_{period}'] = (data['close'] - data['low'].rolling(window=period, min_periods=1).min()) / (
                    data['range'] + epsilon)

    # Полосы Боллинджера (ширина)
    for period in [20]:
        sma = data['close'].rolling(window=period, min_periods=1).mean()
        std = data['close'].rolling(window=period, min_periods=1).std()
        data[f'bb_width_{period}'] = (sma + 2 * std - (sma - 2 * std)) / (sma + epsilon)

    # Список фичей, которые будем лагировать
    base_features = [
        'TR', 'range', 'return', 'abs_return', 'return_squared',
        'body', 'upper_shadow', 'lower_shadow', 'body_to_range',
        'shadow_to_body', 'close_position', 'gap', 'abs_gap'
    ]

    if include_volume:
        base_features.extend(['log_volume', 'volume_TR', 'volume_range'])

    if add_rolling:
        rolling_features = [
            'ATR_5', 'ATR_10', 'ATR_20',
            'TR_vs_ATR_5', 'TR_vs_ATR_10', 'TR_vs_ATR_20',
            'return_std_5', 'return_std_10', 'return_std_20',
            'abs_return_sma_5', 'abs_return_sma_10', 'abs_return_sma_20',
            'dist_from_high_10', 'dist_from_high_20',
            'dist_from_low_10', 'dist_from_low_20',
            'bb_width_20'
        ]
        if include_volume:
            rolling_features.extend(['volume_sma_5', 'volume_sma_10', 'volume_sma_20',
                                     'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20'])
        base_features.extend(rolling_features)

    # Убираем NaN в начале (из-за скользящих окон и лагов)
    data = data.dropna().reset_index(drop=True)

    # === 4. Создаём плоское окно с лагами ===
    X_rows = []
    y_rows = []

    # Минимальное количество данных для одной строки
    min_rows_needed = window_in + window_out

    for i in range(len(data) - min_rows_needed + 1):
        # Входное окно: i : i+window_in
        window = data.iloc[i: i + window_in]

        # Таргет: следующие window_out значений TR
        target = data['TR'].iloc[i + window_in: i + window_in + window_out].values

        # Проверяем, что таргет полный (нет NaN)
        if len(target) == window_out and not np.any(np.isnan(target)):
            # Собираем фичи: для каждой базовой фичи берём все window_in значений
            row = []
            for feat in base_features:
                if feat in data.columns:
                    values = window[feat].values
                    # Дополнительно проверяем на NaN
                    if np.any(np.isnan(values)):
                        break
                    row.extend(values)
            else:  # выполнится, если не было break (все фичи ок)
                X_rows.append(row)
                y_rows.append(target)

    # Создаём имена колонок
    feature_names = []
    for feat in base_features:
        if feat in data.columns:
            for lag in range(window_in):
                feature_names.append(f'{feat}_lag{window_in - 1 - lag}')  # lag0 - самая свежая

    # Преобразуем в numpy массивы
    X = np.array(X_rows)
    y = np.array(y_rows)

    print(f"Создано {X.shape[0]} примеров")
    print(f"Вход: {X.shape[1]} фичей")
    print(f"Выход: {y.shape[1]} шагов")

    return X, y, feature_names


def prepare_single_window_features(
        df_window: pd.DataFrame,
        include_volume: bool = True,
        add_rolling: bool = True,
        epsilon: float = 1e-10
) -> np.ndarray:
    """
    Преобразует ОДИН датафрейм с 20 свечами в фичи для предсказания.

    Parameters:
    -----------
    df_window : pd.DataFrame
        Датафрейм с 20 строками и колонками ['open', 'high', 'low', 'close', 'volume']
    include_volume : bool
        Использовать ли объём в фичах
    add_rolling : bool
        Добавлять ли скользящие статистики (требует больше данных внутри окна)
    epsilon : float
        Для защиты от деления на ноль

    Returns:
    --------
    np.ndarray
        Вектор фичей длиной (n_features,)
    """

    # Проверяем, что пришло ровно 20 свечей
    if len(df_window) != 20:
        raise ValueError(f"Ожидается 20 свечей, получено {len(df_window)}")

    # Копируем, чтобы не портить оригинал
    data = df_window.copy()

    # Проверяем наличие необходимых колонок
    required = ['open', 'high', 'low', 'close']
    if include_volume:
        required.append('volume')

    for col in required:
        if col not in data.columns:
            raise ValueError(f"Колонка '{col}' не найдена в DataFrame")

    # === 1. Базовые фичи (расчёт на всём окне) ===

    # True Range
    data['high_low'] = data['high'] - data['low']

    # Для первой свечи нет предыдущей, берём её же (или 0)
    prev_close = data['close'].shift(1)
    prev_close.iloc[0] = data['close'].iloc[0]  # для первой свечи

    data['high_close_prev'] = abs(data['high'] - prev_close)
    data['low_close_prev'] = abs(data['low'] - prev_close)
    data['TR'] = data[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)

    # Простой диапазон
    data['range'] = data['high'] - data['low']

    # Доходности
    data['return'] = np.log(data['close'] / data['close'].shift(1))
    data['return'].iloc[0] = 0  # для первой свечи
    data['abs_return'] = abs(data['return'])
    data['return_squared'] = data['return'] ** 2

    # Структура свечи
    data['body'] = abs(data['close'] - data['open'])
    data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
    data['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']
    data['body_to_range'] = data['body'] / (data['range'] + epsilon)
    data['shadow_to_body'] = (data['upper_shadow'] + data['lower_shadow']) / (data['body'] + epsilon)

    # Позиция закрытия
    data['close_position'] = (data['close'] - data['low']) / (data['range'] + epsilon)

    # Гэпы
    data['gap'] = data['open'] - data['close'].shift(1)
    data['gap'].iloc[0] = 0
    data['abs_gap'] = abs(data['gap'])

    # === 2. Объёмные фичи ===
    if include_volume:
        data['log_volume'] = np.log(data['volume'] + 1)
        data['volume_TR'] = data['volume'] * data['TR']
        data['volume_range'] = data['volume'] * data['range']

    # === 3. Скользящие статистики (на маленьком окне — приближённо) ===
    if add_rolling:
        # Используем expanding вместо rolling, так как окно маленькое
        # ATR-like
        data['ATR_5'] = data['TR'].rolling(window=min(5, len(data)), min_periods=1).mean()
        data['ATR_10'] = data['TR'].rolling(window=min(10, len(data)), min_periods=1).mean()
        data['ATR_20'] = data['TR'].rolling(window=min(20, len(data)), min_periods=1).mean()

        data['TR_vs_ATR_5'] = data['TR'] / (data['ATR_5'] + epsilon)
        data['TR_vs_ATR_10'] = data['TR'] / (data['ATR_10'] + epsilon)
        data['TR_vs_ATR_20'] = data['TR'] / (data['ATR_20'] + epsilon)

        # Стандартное отклонение доходностей
        data['return_std_5'] = data['return'].rolling(window=min(5, len(data)), min_periods=1).std()
        data['return_std_10'] = data['return'].rolling(window=min(10, len(data)), min_periods=1).std()
        data['return_std_20'] = data['return'].rolling(window=min(20, len(data)), min_periods=1).std()

        data['abs_return_sma_5'] = data['abs_return'].rolling(window=min(5, len(data)), min_periods=1).mean()
        data['abs_return_sma_10'] = data['abs_return'].rolling(window=min(10, len(data)), min_periods=1).mean()
        data['abs_return_sma_20'] = data['abs_return'].rolling(window=min(20, len(data)), min_periods=1).mean()

    # Объёмные скользящие
    if include_volume:
        data['volume_sma_5'] = data['volume'].rolling(window=min(5, len(data)), min_periods=1).mean()
        data['volume_sma_10'] = data['volume'].rolling(window=min(10, len(data)), min_periods=1).mean()
        data['volume_sma_20'] = data['volume'].rolling(window=min(20, len(data)), min_periods=1).mean()

        data['volume_ratio_5'] = data['volume'] / (data['volume_sma_5'] + epsilon)
        data['volume_ratio_10'] = data['volume'] / (data['volume_sma_10'] + epsilon)
        data['volume_ratio_20'] = data['volume'] / (data['volume_sma_20'] + epsilon)

    # Макро-состояние (на доступных данных)
    for period in [10, 20]:
        if period <= len(data):
            data[f'dist_from_high_{period}'] = (data['high'].rolling(window=period, min_periods=1).max() - data[
                'close']) / (data['range'] + epsilon)
            data[f'dist_from_low_{period}'] = (data['close'] - data['low'].rolling(window=period, min_periods=1).min()) / (
                        data['range'] + epsilon)
        else:
            data[f'dist_from_high_{period}'] = 0
            data[f'dist_from_low_{period}'] = 0

    # Полосы Боллинджера
    if len(data) >= 20:
        sma = data['close'].rolling(window=20, min_periods=1).mean()
        std = data['close'].rolling(window=20, min_periods=1).std()
        data['bb_width_20'] = (sma + 2 * std - (sma - 2 * std)) / (sma + epsilon)
    else:
        data['bb_width_20'] = 0

    # === 4. Формируем список фичей в правильном порядке ===
    # ВАЖНО: порядок должен строго соответствовать порядку из обучающей функции!

    base_features = [
        'TR', 'range', 'return', 'abs_return', 'return_squared',
        'body', 'upper_shadow', 'lower_shadow', 'body_to_range',
        'shadow_to_body', 'close_position', 'gap', 'abs_gap'
    ]

    if include_volume:
        base_features.extend(['log_volume', 'volume_TR', 'volume_range'])

    if add_rolling:
        rolling_features = [
            'ATR_5', 'ATR_10', 'ATR_20',
            'TR_vs_ATR_5', 'TR_vs_ATR_10', 'TR_vs_ATR_20',
            'return_std_5', 'return_std_10', 'return_std_20',
            'abs_return_sma_5', 'abs_return_sma_10', 'abs_return_sma_20',
            'dist_from_high_10', 'dist_from_high_20',
            'dist_from_low_10', 'dist_from_low_20',
            'bb_width_20'
        ]
        if include_volume:
            rolling_features.extend(['volume_sma_5', 'volume_sma_10', 'volume_sma_20',
                                     'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20'])
        base_features.extend(rolling_features)

    print(data)
    data = data.fillna(0)
    # Собираем фичи в правильном порядке:
    # Сначала все фичи для lag0 (самая свежая), потом для lag1, и так далее
    feature_vector = []

    for lag in range(19, -1, -1):  # от 19 до 0 (чтобы lag0 был последним/самым свежим)
        for feat in base_features:
            if feat in data.columns:
                # Берём значение для конкретного лага
                # Индекс: -lag-1 означает "lag шагов от конца"
                # lag=0 -> индекс -1 (последний)
                # lag=1 -> индекс -2 (предпоследний)
                val = data[feat].iloc[-lag - 1] if lag < len(data) else 0
                feature_vector.append(val)

    return np.array(feature_vector)


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
        'tick_volume': 'volume'
    }, inplace=True)
    print(f"Загружено {len(df)} баров с {df.index[0]} по {df.index[-1]}")


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
    p = DirectForecaster(model)
    x_data, y_data, features_names = prepare_volatility_features(df, 20, 20)
    p.fit(x_data, y_data, 20)

    print(p.predict(df.iloc[-20:].copy()))