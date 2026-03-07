import joblib
import re
import onnxruntime as ort
import os
from time import sleep
import time as time
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType



class VolClustGB:
    def __init__(self):
        self.models = []
        self.scaler = StandardScaler()
        self.X_shape = 0
        self.is_fitted = False
        self.onnx_load = False
        self.base_model = GradientBoostingRegressor(
            loss='squared_error',
            learning_rate=0.01,
            n_estimators=1,
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42,
            warm_start=True
        )

    def __FichEngine(self,
            df: pd.DataFrame,
            window_in: int,
            window_out: int,
            include_volume: bool = True,
            add_rolling: bool = True,
            epsilon: float = 1e-10
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            data[f'dist_from_high_{period}'] = (data['high'].rolling(window=period, min_periods=1).max() - data[
                'close']) / (
                                                       data['range'] + epsilon)
            data[f'dist_from_low_{period}'] = (data['close'] - data['low'].rolling(window=period,
                                                                                   min_periods=1).min()) / (
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

        #return X, y, feature_names
        return X, y

    def __prepare_single_window_features(self,
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
                data[f'dist_from_low_{period}'] = (data['close'] - data['low'].rolling(window=period,
                                                                                       min_periods=1).min()) / (
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
        feature_vector = []

        for lag in range(19, -1, -1):  # от 19 до 0 (чтобы lag0 был последним/самым свежим)
            for feat in base_features:
                if feat in data.columns:
                    val = data[feat].iloc[-lag - 1] if lag < len(data) else 0
                    feature_vector.append(val)

        return np.array(feature_vector)

    def fit(self, data, input_bars, horizons, trees_count, show_results=False):
        x, y = self.__FichEngine(data, input_bars, horizons)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=42)
        X_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.X_shape = X_scaled.shape[1]
        print('self.X_shape: ', self.X_shape)
        train_errors = []
        val_errors = []
        start = time.time()
        for i in range(1, trees_count+1):
            print(f'{i} trees')
            t_error = 0
            v_error = 0
            # Определяем горизонты
            if isinstance(horizons, int):
                horizons_list = list(range(horizons))
            else:
                horizons_list = horizons

            # Если y - 2D (уже с горизонтами)
            if len(y_train.shape) == 2 and y_train.shape[1] > 1:
                for h_idx, h in enumerate(horizons_list):
                    if h_idx >= y_train.shape[1]:
                        print(f"Предупреждение: горизонт {h} выходит за пределы y, пропускаем")
                        continue

                    # Берем конкретную колонку для этого горизонта
                    y_h = y_train.iloc[:, h_idx] if hasattr(y_train, 'iloc') else y_train[:, h_idx]

                    # Убираем NaN
                    valid_mask = ~pd.isna(y_h) if hasattr(y_h, 'isna') else ~np.isnan(y_h)
                    X_h = X_scaled[valid_mask]
                    y_h_clean = y_h[valid_mask]

                    # Обучаем модель
                    if i != 1:
                        self.models[h_idx].set_params(n_estimators=i)
                        self.models[h_idx].fit(X_h, y_h_clean)
                    else:
                        model = self.base_model.__class__(**self.base_model.get_params())
                        model.n_estimators = i
                        model.fit(X_h, y_h_clean)
                        self.models.append(model)

                    y_h_v = y_test.iloc[:, h_idx] if hasattr(y_test, 'iloc') else y_test[:, h_idx]

                    # Убираем NaN
                    valid_mask = ~pd.isna(y_h_v) if hasattr(y_h_v, 'isna') else ~np.isnan(y_h_v)
                    X_h_v = X_test_scaled[valid_mask]
                    y_h_v_clean = y_h_v[valid_mask]
                    if i != 1:
                        t_error += mean_squared_error(y_h_clean, self.models[h_idx].predict(X_h))
                        v_error += mean_squared_error(y_h_v_clean, self.models[h_idx].predict(X_h_v))
                    else:
                        t_error += mean_squared_error(y_h_clean, model.predict(X_h))
                        v_error += mean_squared_error(y_h_v_clean, model.predict(X_h_v))


            train_errors.append(float(t_error)/horizons)
            val_errors.append(float(v_error)/horizons)
            print(f"Затрачено времени: {time.time() - start} сек")

        if show_results:
            plt.figure(figsize=(10, 6))
            plt.plot(list(train_errors), label='Train Loss')
            plt.plot(list(val_errors), label='Validation Loss')
            plt.xlabel('Trees')
            plt.ylabel('MSE Loss')
            plt.title('Training and Test Loss over Trees')
            plt.legend()
            plt.grid(True)
            plt.show()

        print('model is trained')
        self.is_fitted = True

    def forecast(self, latest_data):
        X = self.__prepare_single_window_features(latest_data)
        if not self.is_fitted and not self.onnx_load:
            raise ValueError("Модель еще не обучена!")

        if self.onnx_load:
            predictions = []

            # Преобразуем входные данные в float32 (ONNX требует float32)
            X_float32 = X.astype(np.float32)
            if len(X_float32.shape) == 1:
                # Добавляем размерность батча: (n_features,) -> (1, n_features)
                X_float32 = X_float32.reshape(1, -1)

            X_float32 = self.scaler.transform(X_float32)

            for i, session in enumerate(self.loaded_models):
                # Получаем имя входного тензора
                input_name = session.get_inputs()[0].name

                # Делаем предсказание
                pred = session.run(None, {input_name: X_float32})
                predictions.append(float(f'{pred[0][0][0]:.7f}'))

                print(f"Модель {i} сделала предсказание")

            return np.array(predictions)
        else:
            # Масштабируем X
            X = X.astype(np.float32)
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


    def save(self, name):

        # Определяем тип входных данных: [None, 5] означает, что batch size может быть любым,
        # а каждая "строка" (сэмпл) состоит из 5 признаков.
        os.makedirs(name, exist_ok=True)
        initial_type = [('float_input', FloatTensorType([None, self.X_shape]))]

        if hasattr(self, 'scaler'):
            scaler_path = os.path.join(name, f"{name}_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            print(f"Скалер сохранён: {scaler_path}")

        for i in range(len(self.models)):
            onx = convert_sklearn(self.models[i], initial_types=initial_type, target_opset=12)

            # Формируем путь к файлу
            file_path = os.path.join(name, f"{name}_{i}.onnx")

            # Сохраняем файл
            with open(file_path, "wb") as f:
                f.write(onx.SerializeToString())


    def load(self, name):
        self.loaded_models = []

        if not os.path.exists(name):
            raise FileNotFoundError(f"Папка {name} не найдена")

        scaler_files = [f for f in os.listdir(name) if f.endswith('_scaler.pkl')]
        if scaler_files:
            scaler_path = os.path.join(name, scaler_files[0])
            self.scaler = joblib.load(scaler_path)
            print(f"Скалер загружен: {scaler_files[0]}")
            print(f"  mean: {self.scaler.mean_}")
            print(f"  scale: {self.scaler.scale_}")


            # Получаем все .onnx файлы в папке
        model_files = [f for f in os.listdir(name) if f.endswith('.onnx')]

        if not model_files:
            raise FileNotFoundError(f"В папке {name} не найдено .onnx файлов")

        # Сортируем файлы для consistent порядка
        print(model_files)
        model_files.sort()
        ml = len(model_files)
        numbers = {}
        for f in model_files:
            match = re.search(r'_(\d+)\.onnx$', f)
            if match:
                num = int(match.group(1))
                numbers[num] = f

        model_files = []
        for i in range(ml):
            model_files.append(numbers[i])

        print(model_files)

        print(f"Найдено моделей: {len(model_files)}")

        # Загружаем каждую модель
        for model_file in model_files:
            model_path = os.path.join(name, model_file)

            # Создаем сессию ONNX Runtime
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.loaded_models.append(session)

            print(f"Загружена модель: {model_file}")
            print(f"  Входные данные: {[inp.name for inp in session.get_inputs()]}")
            print(f"  Выходные данные: {[out.name for out in session.get_outputs()]}")

        self.onnx_load = True
