import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
import re
import onnxruntime as ort
import os
import onnxmltools
from onnxconverter_common.data_types import FloatTensorType
from .visual import Visualization
import time as time
import numpy as np
import xgboost
import lightgbm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')


class FichEn:
    def _DataSplitting(self,
            df: pd.DataFrame,
            window_in: int,
            window_out: int,
            include_volume: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Копируем, чтобы не портить оригинал
        data = df.copy()

        # Проверка колонок
        required = ['open', 'high', 'low', 'close']
        if include_volume:
            required.append('volume')

        for col in required:
            if col not in data.columns:
                raise ValueError(f"Колонка '{col}' не найдена")

        # === 1. Сначала создаем "сырые" окна ===
        raw_windows_X = []  # список окон для входных данных
        raw_windows_y = []  # список окон для целевых значений

        # Правильные индексы:
        # от window_in до len(data)-window_out
        for i in range(window_in, len(data) - window_out + 1):
            # Входное окно: window_in свечей ДО момента i
            x_window = data.iloc[i - window_in:i]

            # Выходное окно: window_out свечей ПОСЛЕ момента i
            y_window = data.iloc[i:i + window_out]

            raw_windows_X.append(x_window)
            raw_windows_y.append(y_window)

        # Преобразуем в numpy массивы
        X = raw_windows_X
        y = raw_windows_y

        #print(f"Создано {X.shape[0]} примеров")
        #print(f"Вход: {X.shape[1]} фичей")
        #print(f"Выход: {y.shape[1]} шагов")

        #return X, y, feature_names
        return X, y

    def _prepare_single_window_features(self, df_window: pd.DataFrame, epsilon: float = 1e-10) -> np.ndarray:
        # Копируем и сбрасываем индекс
        data = df_window.copy().reset_index(drop=True)

        # Проверка колонок
        required = ['open', 'high', 'low', 'close']

        for col in required:
            if col not in data.columns:
                raise ValueError(f"Колонка '{col}' не найдена")

        close_price = data['close'].values

        # === 1. БАЗОВЫЕ ФИЧИ (самое важное для волатильности) ===

        # TR и его компоненты
        data['high_low'] = (data['high'] - data['low']) / (close_price + epsilon)
        prev_close = data['close'].shift(1).fillna(data['close'])
        data['TR'] = np.maximum(
            data['high_low'],
            np.maximum(
                abs(data['high'] - prev_close) / (close_price + epsilon),
                abs(data['low'] - prev_close) / (close_price + epsilon)
            )
        )

        # Доходности (основа волатильности)
        data['return'] = np.log(data['close'] / data['close'].shift(1)).fillna(0)
        data['abs_return'] = data['return'].abs()

        # Гэпы (важно для волатильности)
        data['gap'] = (data['open'] - data['close'].shift(1)).fillna(0) / (close_price + epsilon)

        # Структура свечи (упрощенно)
        data['body'] = abs(data['close'] - data['open']) / (close_price + epsilon)
        data['shadow'] = (data['high'] - data[['open', 'close']].max(axis=1) +
                          (data[['open', 'close']].min(axis=1) - data['low'])) / (close_price + epsilon)

        # Позиция в свече (информативно для разворота волатильности)
        data['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + epsilon)

        # === 4. ФИНАЛЬНЫЙ СПИСОК ФИЧЕЙ (ОПТИМИЗИРОВАННЫЙ) ===
        base_features = [
            'TR',  # базовая волатильность
            'return',  # доходность
            'abs_return',  # абсолютная доходность
            'gap',  # гэпы
            'body',  # тело свечи
            'shadow',  # тени
            'close_position'  # позиция закрытия
        ]

        # Оставляем только существующие колонки
        existing_features = [f for f in base_features if f in data.columns]

        # Заполняем NaN
        data = data.fillna(0)

        # Собираем фичи (берем ВСЕ свечи в окне)
        feature_vector = []
        for i in range(len(data)):
            for feat in existing_features:
                feature_vector.append(data[feat].iloc[i])

        return np.array(feature_vector)


    def _prepare_single_window_target(self, df_window: pd.DataFrame) -> np.ndarray:
        """
        Возвращает массив True Range для каждой свечи в окне
        """
        data = df_window.copy()

        required = ['open', 'high', 'low', 'close']

        for col in required:
            if col not in data.columns:
                raise ValueError(f"Колонка '{col}' не найдена")

        # Расчет TR для каждой свечи
        tr_values = []

        for i in range(len(data)):
            if i == 0:
                # Для первой свечи используем только high-low
                tr = (data['high'].iloc[i] - data['low'].iloc[i]) / data['close'].iloc[i]
            else:
                hl = data['high'].iloc[i] - data['low'].iloc[i]
                hc = abs(data['high'].iloc[i] - data['close'].iloc[i - 1])
                lc = abs(data['low'].iloc[i] - data['close'].iloc[i - 1])
                tr = max(hl/data['close'].iloc[i], hc/data['close'].iloc[i], lc/data['close'].iloc[i])

            tr_values.append(tr)

        return np.array(tr_values)


    def quick_correlation_analysis(self, X, threshold=0.95):
        """
        Быстрый анализ корреляций без VIF (для большого числа фичей)
        """
        n_features = X.shape[1]
        print(f"Анализ корреляций для {n_features} фичей")

        # 1. Удаляем фичи с нулевой дисперсией
        std_devs = X.std(axis=0)
        constant_features = np.where(std_devs < 1e-10)[0]

        if len(constant_features) > 0:
            print(f"Найдено {len(constant_features)} фичей с нулевой дисперсией (удаляем из анализа)")
            # Удаляем константные фичи
            X_clean = X[:, std_devs >= 1e-10]
            n_features_clean = X_clean.shape[1]
        else:
            X_clean = X
            n_features_clean = n_features

        # 2. Стандартизация с защитой от нулевой дисперсии
        X_std = (X_clean - X_clean.mean(axis=0)) / (X_clean.std(axis=0) + 1e-10)

        # 3. Быстрый расчет корреляционной матрицы
        corr_matrix = np.corrcoef(X_std.T)

        # 4. Находим высокие корреляции
        high_corr_pairs = []
        for i in range(n_features_clean):
            for j in range(i + 1, n_features_clean):
                corr_val = corr_matrix[i, j]
                if not np.isnan(corr_val) and abs(corr_val) > threshold:
                    high_corr_pairs.append((i, j, corr_val))

        print(f"\nНайдено {len(high_corr_pairs)} пар с корреляцией > {threshold}")

        # 5. Статистика
        if len(high_corr_pairs) > 0:
            corr_values = [p[2] for p in high_corr_pairs]
            print(f"Средняя корреляция в проблемных парах: {np.mean(corr_values):.3f}")
            print(f"Максимальная корреляция: {np.max(corr_values):.3f}")

            # Покажем первые 10 проблемных пар
            print("\nПримеры высококоррелированных пар:")
            for i, (f1, f2, corr) in enumerate(high_corr_pairs[:10]):
                print(f"  Фича {f1} - Фича {f2}: {corr:.3f}")

        # 6. Визуализация
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Гистограмма корреляций (игнорируем NaN)
        corr_flat = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        corr_flat = corr_flat[~np.isnan(corr_flat)]  # убираем NaN

        axes[0].hist(corr_flat, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=threshold, color='r', linestyle='--', label=f'threshold={threshold}')
        axes[0].axvline(x=-threshold, color='r', linestyle='--')
        axes[0].set_xlabel('Корреляция')
        axes[0].set_ylabel('Частота')
        axes[0].set_title(f'Распределение корреляций (всего {len(corr_flat)} пар)')
        axes[0].legend()

        # Тепловая карта
        if n_features_clean > 100:
            # Показываем только первые 100 фичей
            corr_small = corr_matrix[:100, :100]
            im = axes[1].imshow(corr_small, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(im, ax=axes[1])
            axes[1].set_title('Корреляции (первые 100 фичей)')
        else:
            im = axes[1].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(im, ax=axes[1])
            axes[1].set_title('Полная корреляционная матрица')

        plt.tight_layout()
        plt.show()

        return corr_matrix, high_corr_pairs


    def fit(self, data, input_bars, horizon, trees_count, show_results=False, feature_func=None, target_func=None):
        x, y = self._DataSplitting(data, input_bars, horizon, True)
        XX = []
        YY = []
        lx = len(x)
        if len(x) != len(y): raise "pizdec"
        for i in range(len(x)):
            startt = time.time()
            if feature_func == None:
                window_features = self._prepare_single_window_features(x[i])
            else:
                window_features = feature_func(x[i])

            if target_func == None:
                window_targets = self._prepare_single_window_target(y[i])
            else:
                window_targets = target_func(x[i])

            XX.append(window_features)
            YY.append(window_targets)

            percent = (float(i) / lx) * 100
            filled_chars = int((i / lx) * 50)  # Допустим, бар шириной 50 символов
            bar = '█' * filled_chars + '░' * (50 - filled_chars)
            time_per = time.time() - startt
            if percent < 10:
                print(f'\rПодготовка данных: |{bar}|   {percent:.2f}%  Осталось {time_per*(lx-i):.2f} секунд', end='', flush=True)
            elif percent >= 10 and percent < 100:
                print(f'\rПодготовка данных: |{bar}|  {percent:.2f}%   Осталось {time_per*(lx-i):.2f} секунд', end='', flush=True)
            else:
                print(f'\rПодготовка данных: |{bar}| {percent:.2f}%    Осталось {time_per*(lx-i):.2f} секунд', end='', flush=True)

        print()
        x = np.array(XX)
        y = np.array(YY)
        print(f"Создано {x.shape[0]} примеров")
        print(f"Вход: {x.shape[1]} фичей")
        print(f"Выход: {y.shape[1]} шагов")
        self.quick_correlation_analysis(x)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=42)
        X_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.X_shape = X_scaled.shape[1]

        # early stopping variables
        previous_error = 0
        previous_error_was_grow = False

        self.train_errors = []
        self.val_errors = []
        trees = 1
        start = time.time()
        try:
            for i in range(trees, trees_count+1):
                print(f'{i} trees')
                trees = i
                t_error = 0
                v_error = 0
                # Определяем горизонты
                if isinstance(horizon, int):
                    horizon_list = list(range(horizon))
                else:
                    horizon_list = horizon

                # Если y - 2D (уже с горизонтами)
                if len(y_train.shape) == 2 and y_train.shape[1] > 1:
                    for h_idx, h in enumerate(horizon_list):
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


                var_test_error = float(t_error)/horizon
                var_val_error = float(v_error)/horizon

                if self.early_stopping:
                    if previous_error_was_grow:
                        print(f'training stopped by early stopping on {i} trees')
                        if show_results:
                            self.V.show_errors(self.train_errors, self.val_errors)
                        self.is_fitted = True
                        return
                    else:
                        if previous_error != 0 and previous_error < var_val_error:
                            previous_error_was_grow = True

                previous_error = var_val_error

                self.train_errors.append(var_test_error)
                self.val_errors.append(var_val_error)
                print('Train error:      ', var_test_error)
                print('Validation error: ', var_val_error)
                print(f"Затрачено времени: {time.time() - start} сек")

        except KeyboardInterrupt:
            print("\nОбучение прервано по Ctrl+C!")

        if show_results:
            self.V.show_errors(self.train_errors, self.val_errors)

        print('model is trained')
        self.is_fitted = True

    def forecast(self, latest_data, feature_func=None, show=False):
        if feature_func == None:
            X = self._prepare_single_window_features(latest_data)
        else:
            X = feature_func(latest_data)

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

            if show:
                epsilon = 1e-10
                close_price = latest_data['close'].values
                latest_data['high_low'] = (latest_data['high'] - latest_data['low']) / (close_price + epsilon)
                prev_close = latest_data['close'].shift(1).fillna(latest_data['close'])
                latest_data['TR'] = np.maximum(
                    latest_data['high_low'],
                    np.maximum(
                        abs(latest_data['high'] - prev_close) / (close_price + epsilon),
                        abs(latest_data['low'] - prev_close) / (close_price + epsilon)
                    )
                )
                rez = np.concatenate([np.array(latest_data['TR'].values), np.array(predictions)])
                dates = pd.date_range(start='2024-01-01', periods=len(rez), freq='D')
                rez = pd.DataFrame(rez, index=dates, columns=['value'])
                self.V.show_vol(rez, len(predictions))
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

            if show:
                epsilon = 1e-10
                close_price = latest_data['close'].values
                latest_data['high_low'] = (latest_data['high'] - latest_data['low']) / (close_price + epsilon)
                prev_close = latest_data['close'].shift(1).fillna(latest_data['close'])
                latest_data['TR'] = np.maximum(
                    latest_data['high_low'],
                    np.maximum(
                        abs(latest_data['high'] - prev_close) / (close_price + epsilon),
                        abs(latest_data['low'] - prev_close) / (close_price + epsilon)
                    )
                )
                rez = np.concatenate([np.array(latest_data['TR'].values), np.array(predictions)])
                dates = pd.date_range(start='2024-01-01', periods=len(rez), freq='D')
                rez = pd.DataFrame(rez, index=dates, columns=['value'])
                self.V.show_vol(rez, len(predictions))
            return np.array(predictions)


    def show_train_results(self):
        self.V.show_errors(self.train_errors, self.val_errors)



class VolClustGB(FichEn):
    def __init__(self, sett, default=True, early_stopping=True):
        self.models = []
        self.scaler = StandardScaler()
        self.X_shape = 0
        self.is_fitted = False
        self.onnx_load = False
        self.early_stopping = early_stopping
        self.V = Visualization('dark')
        if default:
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
        else:
            self.base_model = GradientBoostingRegressor(**sett)


    def save(self, name):

        # Определяем тип входных данных: [None, 5] означает, что batch size может быть любым,
        # а каждая "строка" (сэмпл) состоит из 5 признаков.
        os.makedirs(name, exist_ok=True)
        initial_type = [('float_input', FloatTensorType([None, self.X_shape]))]

        if hasattr(self, 'scaler'):
            scaler_path = os.path.join(name, f"{name}_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            #print(f"Скалер сохранён: {scaler_path}")

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

            # Получаем все .onnx файлы в папке
        model_files = [f for f in os.listdir(name) if f.endswith('.onnx')]

        if not model_files:
            raise FileNotFoundError(f"В папке {name} не найдено .onnx файлов")

        # Сортируем файлы для consistent порядка
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


        #print(f"Найдено моделей: {len(model_files)}")

        # Загружаем каждую модель
        for model_file in model_files:
            model_path = os.path.join(name, model_file)
            # Создаем сессию ONNX Runtime
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

            input_info = session.get_inputs()[0]  # берем первый вход

            shape = input_info.shape

            self.loaded_models.append(session)

        self.onnx_load = True


class VolClustXGB(FichEn):
    def __init__(self, sett, default=True, early_stopping=True):
        self.models = []
        self.scaler = StandardScaler()
        self.X_shape = 0
        self.is_fitted = False
        self.onnx_load = False
        self.early_stopping = early_stopping
        self.V = Visualization('dark')
        if default:
            self.base_model = xgboost.XGBRegressor(
                objective='reg:squarederror',  # вместо 'loss'
                learning_rate=0.01,
                n_estimators=1,
                max_depth=3,
                min_child_weight=5,
                subsample=0.8,
                random_state=42,
            )
        else:
            self.base_model = xgboost.XGBRegressor(**sett)


    def save(self, name):

        # Определяем тип входных данных: [None, 5] означает, что batch size может быть любым,
        # а каждая "строка" (сэмпл) состоит из 5 признаков.
        os.makedirs(name, exist_ok=True)
        initial_type = [('float_input', FloatTensorType([None, self.X_shape]))]

        if hasattr(self, 'scaler'):
            scaler_path = os.path.join(name, f"{name}_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            #print(f"Скалер сохранён: {scaler_path}")

        for i in range(len(self.models)):
            onx = onnxmltools.convert_xgboost(self.models[i], initial_types=initial_type, target_opset=12)

            # Формируем путь к файлу
            file_path = os.path.join(name, f"{name}_{i}.onnx")

            onnxmltools.utils.save_model(onx, file_path)



    def load(self, name):
        self.loaded_models = []

        if not os.path.exists(name):
            raise FileNotFoundError(f"Папка {name} не найдена")

        scaler_files = [f for f in os.listdir(name) if f.endswith('_scaler.pkl')]
        if scaler_files:
            scaler_path = os.path.join(name, scaler_files[0])
            self.scaler = joblib.load(scaler_path)

            # Получаем все .onnx файлы в папке
        model_files = [f for f in os.listdir(name) if f.endswith('.onnx')]

        if not model_files:
            raise FileNotFoundError(f"В папке {name} не найдено .onnx файлов")

        # Сортируем файлы для consistent порядка
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


        #print(f"Найдено моделей: {len(model_files)}")

        # Загружаем каждую модель
        for model_file in model_files:
            model_path = os.path.join(name, model_file)
            # Создаем сессию ONNX Runtime
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

            input_info = session.get_inputs()[0]  # берем первый вход

            shape = input_info.shape

            self.loaded_models.append(session)

        self.onnx_load = True


class VolClustLightGB(FichEn):
    def __init__(self, sett, default=True, early_stopping=True):
        self.models = []
        self.scaler = StandardScaler()
        self.X_shape = 0
        self.is_fitted = False
        self.onnx_load = False
        self.early_stopping = early_stopping
        self.V = Visualization('dark')
        if default:
            self.base_model = lightgbm.LGBMRegressor(
                objective='regression',
                learning_rate=0.01,
                n_estimators=1,
                max_depth=3,
                min_sum_hessian_in_leaf=5,  # вместо min_child_weight
                subsample=0.8,  # или bagging_fraction
                subsample_freq=1,  # частота применения subsample
                random_state=42,
                verbose=-1
            )
        else:
            self.base_model = lightgbm.LGBMRegressor(**sett)


    def save(self, name):

        # Определяем тип входных данных: [None, 5] означает, что batch size может быть любым,
        # а каждая "строка" (сэмпл) состоит из 5 признаков.
        os.makedirs(name, exist_ok=True)
        initial_type = [('float_input', FloatTensorType([None, self.X_shape]))]

        if hasattr(self, 'scaler'):
            scaler_path = os.path.join(name, f"{name}_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            #print(f"Скалер сохранён: {scaler_path}")

        for i in range(len(self.models)):
            #onx = convert_sklearn(self.models[i], initial_types=initial_type, target_opset=12)
            onx = onnxmltools.convert_lightgbm(self.models[i], initial_types=initial_type, zipmap=False, target_opset=12)

            # Формируем путь к файлу
            file_path = os.path.join(name, f"{name}_{i}.onnx")

            onnxmltools.utils.save_model(onx, file_path)


    def load(self, name):
        self.loaded_models = []

        if not os.path.exists(name):
            raise FileNotFoundError(f"Папка {name} не найдена")

        scaler_files = [f for f in os.listdir(name) if f.endswith('_scaler.pkl')]
        if scaler_files:
            scaler_path = os.path.join(name, scaler_files[0])
            self.scaler = joblib.load(scaler_path)

            # Получаем все .onnx файлы в папке
        model_files = [f for f in os.listdir(name) if f.endswith('.onnx')]

        if not model_files:
            raise FileNotFoundError(f"В папке {name} не найдено .onnx файлов")

        # Сортируем файлы для consistent порядка
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


        # Загружаем каждую модель
        for model_file in model_files:
            model_path = os.path.join(name, model_file)
            # Создаем сессию ONNX Runtime
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

            input_info = session.get_inputs()[0]  # берем первый вход

            shape = input_info.shape

            self.loaded_models.append(session)

        self.onnx_load = True
