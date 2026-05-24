import json
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

import lightgbm as lgb



class FichEn:
    def _DataSplitting(self,
            df: pd.DataFrame,
            window_in: int,
            window_out: int,
            include_volume: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = df.copy()
        self.input_bars = window_in
        required = ['open', 'high', 'low', 'close']
        if include_volume:
            required.append('volume')

        for col in required:
            if col not in data.columns:
                raise ValueError(f"Колонка '{col}' не найдена")

        raw_windows_X = []
        raw_windows_y = []

        for i in range(window_in + 1, len(data) - window_out + 1):
            x_window = data.iloc[i - window_in: i]
            y_window = data.iloc[i - 1: i + window_out]

            raw_windows_X.append(x_window)
            raw_windows_y.append(y_window)

        X = raw_windows_X
        y = raw_windows_y

        return X, y

    def _prepare_single_window_features(self, df_window: pd.DataFrame, feature_list, epsilon: float = 1e-10) -> np.ndarray:
        data = df_window.copy().reset_index(drop=True)

        required = ['open', 'high', 'low', 'close']

        for col in required:
            if col not in data.columns:
                raise ValueError(f"Колонка '{col}' не найдена")

        feature_list_final = []
        single_feature_list_final = []
        single_feature_dict = {}

        # Сначала рассчитываем TR, так как он нужен для ATR
        if 'TR' in feature_list or any(f.startswith('atr_') for f in feature_list) or any(
                f.startswith('roll_atr_') for f in feature_list):
            prev_close = data['close'].shift(1).fillna(data['close'])
            data['high_low'] = (data['high'] - data['low']) / (data['close'] + epsilon)
            data['TR'] = np.maximum(
                data['high_low'],
                np.maximum(
                    abs(data['high'] - prev_close) / (data['close'] + epsilon),
                    abs(data['low'] - prev_close) / (data['close'] + epsilon)
                )
            )

        for i in feature_list:
            if i == 'high_low':
                feature_list_final.append(i)

            elif i == 'TR':
                feature_list_final.append(i)

            elif i == 'parkinson':
                data[i] = np.sqrt(
                    (1 / (4 * np.log(2))) * (np.log(data['high'] / data['low'])) ** 2
                )
                feature_list_final.append(i)

            elif i == 'garman_klass':
                hl = np.log(data['high'] / data['low']) ** 2
                co = np.log(data['close'] / data['open']) ** 2
                data[i] = np.sqrt(0.5 * hl - (2 * np.log(2) - 1) * co)
                feature_list_final.append(i)

            elif i == 'rogers_satchell':
                h_o = np.log(data['high'] / data['open'])
                l_o = np.log(data['low'] / data['open'])
                c_o = np.log(data['close'] / data['open'])
                data[i] = np.sqrt(h_o * (h_o - c_o) + l_o * (l_o - c_o))
                feature_list_final.append(i)

            elif i == 'returns':
                data[i] = np.log(data['close'] / data['close'].shift(1)).fillna(0)
                feature_list_final.append(i)

            elif i == 'abs_returns':
                # Используем уже рассчитанные returns
                if 'returns' in data.columns:
                    data[i] = data['returns'].abs()
                else:
                    data[i] = np.log(data['close'] / data['close'].shift(1)).fillna(0).abs()
                feature_list_final.append(i)

            elif i == 'gap':
                data[i] = (data['open'] - data['close'].shift(1)).fillna(0) / (data['close'] + epsilon)
                feature_list_final.append(i)

            elif i == 'body':
                data[i] = abs(data['close'] - data['open']) / (data['close'] + epsilon)
                feature_list_final.append(i)

            elif i == 'shadow':
                data[i] = (data['high'] - data[['open', 'close']].max(axis=1) +
                                  (data[['open', 'close']].min(axis=1) - data['low'])) / (data['close'] + epsilon)
                feature_list_final.append(i)

            elif i == 'close_position':
                high_low_diff = data['high'] - data['low']
                data[i] = np.where(high_low_diff > epsilon, (data['close'] - data['low']) / high_low_diff,0.5)
                feature_list_final.append(i)

            elif i == 'month':
                last_datetime = data['time'].iloc[-1]
                single_feature_dict[i] = last_datetime.month
                single_feature_list_final.append(i)

            elif i == 'day_of_month':
                last_datetime = data['time'].iloc[-1]
                single_feature_dict[i] = last_datetime.day
                single_feature_list_final.append(i)

            elif i == 'day_of_week':
                last_datetime = data['time'].iloc[-1]
                single_feature_dict[i] = last_datetime.weekday()+1
                single_feature_list_final.append(i)

            elif i == 'hour':
                last_datetime = data['time'].iloc[-1]
                single_feature_dict[i] = last_datetime.hour
                single_feature_list_final.append(i)

            elif i == 'roll_month':
                last_datetime = data['time']
                data[i] = last_datetime.dt.month
                feature_list_final.append(i)

            elif i == 'roll_day_of_month':
                last_datetime = data['time']
                data[i] = last_datetime.dt.day
                feature_list_final.append(i)

            elif i == 'roll_day_of_week':
                last_datetime = data['time']
                data[i] = last_datetime.dt.weekday+1
                feature_list_final.append(i)

            elif i == 'roll_hour':
                last_datetime = data['time']
                data[i] = last_datetime.dt.hour
                feature_list_final.append(i)

            elif i[:4] == 'rsi_':
                window = ''
                for j in range(4, len(i)):
                    try:
                        int(i[j])
                    except ValueError:
                        break
                    else:
                        window += i[j]
                window = int(window)
                delta = data['close'].diff().iloc[-window:]
                if window > len(delta):
                    raise Exception(
                        f'Not enough data to calculate RSI: max RSI period is {len(delta)} and yours {window}')
                gain = delta.where(delta > 0, 0).mean()
                loss = (-delta.where(delta < 0, 0)).mean()
                rs = gain / (loss + epsilon)
                single_feature_dict[f'rsi_{window}'] = 100 - (100 / (1 + rs))
                single_feature_list_final.append(f'rsi_{window}')

            elif i[:4] == 'atr_':
                window = ''
                for j in range(4, len(i)):
                    try:
                        int(i[j])
                    except ValueError:
                        break
                    else:
                        window += i[j]
                window = int(window)
                if 'TR' not in data.columns:
                    raise Exception('TR must be calculated before ATR')
                tr_values = data['TR'].iloc[-window:]
                if window > len(tr_values):
                    raise Exception(
                        f'Not enough data to calculate ATR: max ATR period is {len(tr_values)} and yours {window}')
                single_feature_dict[f'atr_{window}'] = tr_values.mean() / (data['close'].iloc[-1] + epsilon)
                single_feature_list_final.append(f'atr_{window}')

            elif i[:3] == 'bb_':
                window = ''
                for j in range(3, len(i)):
                    try:
                        int(i[j])
                    except ValueError:
                        break
                    else:
                        window += i[j]
                window = int(window)
                close_values = data['close'].iloc[-window:]
                if window > len(close_values):
                    raise Exception(
                        f'Not enough data to calculate BB: max BB period is {len(close_values)} and yours {window}')
                sma = close_values.mean()
                std = close_values.std()
                single_feature_dict[f'bb_{window}'] = (4 * std) / (sma + epsilon)  # упрощено
                single_feature_list_final.append(f'bb_{window}')

            elif i[:9] == 'roll_rsi_':
                window = ''
                for j in range(9, len(i)):
                    try:
                        int(i[j])
                    except ValueError:
                        break
                    else:
                        window += i[j]
                window = int(window)
                delta = data['close'].diff()
                if window > len(delta):
                    raise Exception(
                        f'Not enough data to calculate RSI: max RSI period is {len(delta)} and yours {window}')
                gain = delta.where(delta > 0, 0).rolling(window, min_periods=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window, min_periods=window).mean()
                rs = gain / (loss + epsilon)
                data[f'roll_rsi_{window}'] = 100 - (100 / (1 + rs))
                feature_list_final.append(f'roll_rsi_{window}')

            elif i[:9] == 'roll_atr_':
                window = ''
                for j in range(9, len(i)):
                    try:
                        int(i[j])
                    except ValueError:
                        break
                    else:
                        window += i[j]
                window = int(window)
                if 'TR' not in data.columns:
                    raise Exception('TR must be calculated before ATR')
                data[f'roll_atr_{window}'] = data['TR'].rolling(window, min_periods=window).mean() / (
                            data['close'] + epsilon)
                feature_list_final.append(f'roll_atr_{window}')

            elif i[:8] == 'roll_bb_':
                window = ''
                for j in range(8, len(i)):
                    try:
                        int(i[j])
                    except ValueError:
                        break
                    else:
                        window += i[j]
                window = int(window)
                sma = data['close'].rolling(window=window, min_periods=window).mean()
                std = data['close'].rolling(window=window, min_periods=window).std()
                data[f'roll_bb_{window}'] = (4 * std) / (sma + epsilon)  # упрощено
                feature_list_final.append(f'roll_bb_{window}')

            else:
                raise ValueError(f'There is not a {i}')


        existing_features = [f for f in feature_list_final if f in data.columns]

        data = data.fillna(0)

        """feature_vector = []
        for i in range(1, len(data)):
            for feat in existing_features:
                feature_vector.append(data[feat].iloc[i])

        for i in single_feature_list_final:
            feature_vector.append(single_feature_dict[i])"""

        feature_vector = data[existing_features].iloc[1:].to_numpy(dtype=np.float64).ravel()
        single_vals = np.array([single_feature_dict[f] for f in single_feature_list_final], dtype=np.float64)
        feature_vector = np.concatenate([feature_vector, single_vals])

        self.roll_features = existing_features
        self.single_features = single_feature_list_final

        return np.array(feature_vector)


    def _prepare_single_window_target(self, df_window: pd.DataFrame) -> np.ndarray:
        data = df_window.copy()

        required = ['open', 'high', 'low', 'close']

        for col in required:
            if col not in data.columns:
                raise ValueError(f"Колонка '{col}' не найдена")

        """tr_values = []

        for i in range(1, len(data)):
            if i == 0:
                tr = (data['high'].iloc[i] - data['low'].iloc[i]) / data['close'].iloc[i]
            else:
                hl = data['high'].iloc[i] - data['low'].iloc[i]
                hc = abs(data['high'].iloc[i] - data['close'].iloc[i - 1])
                lc = abs(data['low'].iloc[i] - data['close'].iloc[i - 1])
                tr = max(hl/data['close'].iloc[i], hc/data['close'].iloc[i], lc/data['close'].iloc[i])

            tr_values.append(tr)"""

        high = data['high']
        low = data['low']
        close = data['close']

        prev_close = close.shift(1)

        hl = high - low
        hc = (high - prev_close).abs()
        lc = (low - prev_close).abs()

        tr_raw = np.maximum(np.maximum(hl, hc), lc)
        tr = tr_raw / close
        tr_values = tr.iloc[1:].to_numpy(dtype=np.float64)

        return np.array(tr_values)

    def forward(self, data, feature_list, trees, input_bars, horizon, trees_count, show_results=False, feature_func=None, target_func=None):
        self.input_bars = input_bars
        self.horizon = horizon
        self.trees_count = trees_count
        self.meta = {
            "input_bars": self.input_bars,
            "horizon": self.horizon,
            "trees_count": self.trees_count
        }
        x, y = self._DataSplitting(data, input_bars, horizon, True)
        XX = []
        YY = []

        lx = len(x)
        self.feature_list = feature_list
        if len(x) != len(y): raise "pizdec"
        for i in range(len(x)):
            startt = time.time()
            if feature_func == None:
                window_features = self._prepare_single_window_features(x[i], feature_list)
            else:
                window_features = feature_func(x[i])

            if target_func == None:
                window_targets = self._prepare_single_window_target(y[i])
            else:
                window_targets = target_func(y[i])

            XX.append(window_features)
            YY.append(window_targets)

            percent = (float(i) / lx) * 100
            filled_chars = int((i / lx) * 50)
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

        print("=" * 60)
        print("ЭТАП 2: WALK-FORWARD ВАЛИДАЦИЯ (ФИКСИРОВАННОЕ ОКНО)")
        print("=" * 60)

        # Параметры фиксированного окна
        train_window_size = input_bars  # сколько последних окон использовать для обучения
        # Минимальный индекс, с которого можно начинать валидацию (должен быть >= train_window_size)
        start_val_idx = train_window_size

        total_iterations = len(x) - start_val_idx
        if total_iterations <= 0:
            raise ValueError(f"Слишком мало данных: всего окон {len(x)}, а train_window_size={train_window_size}")

        print(f"Всего окон: {len(x)}")
        print(f"Размер окна обучения: {train_window_size}")
        print(f"Шагов валидации: {total_iterations}")

        all_train_errors = []
        all_val_errors = []
        all_train_r2 = []
        all_val_r2 = []

        # horizon может быть числом или списком, приведём к списку индексов
        if isinstance(horizon, int):
            horizon_list = list(range(horizon))
        else:
            horizon_list = horizon

        for val_idx in range(start_val_idx, len(x)):
            start_it = time.time()
            iter_num = val_idx - start_val_idx + 1

            # === Обучающая выборка: последние train_window_size окон перед val_idx ===
            train_indices = list(range(val_idx - train_window_size, val_idx))
            X_train = x[train_indices]
            y_train = y[train_indices]

            # === Валидационное окно (одно) ===
            X_val = x[val_idx:val_idx + 1]  # форма (1, n_features)
            y_val_true = y[val_idx]  # форма (horizon,)

            # === Нормализация (обучаем scaler ТОЛЬКО на тренировочных данных) ===
            scaler_X = StandardScaler()
            scaler_y_local = StandardScaler()

            X_train_scaled = scaler_X.fit_transform(X_train)
            y_train_scaled = scaler_y_local.fit_transform(y_train)

            X_val_scaled = scaler_X.transform(X_val)

            # === Обучение моделей для каждого горизонта ===
            model_ex = self.base_model.__class__(**self.base_model.get_params())
            model_ex.set_params(n_estimators=trees)
            models_temp = []
            for h_idx in horizon_list:
                if h_idx >= y_train_scaled.shape[1]:
                    continue  # на всякий случай, если горизонт больше доступного

                y_h = y_train_scaled[:, h_idx]

                # Создаём модель с параметрами, как у self.base_model
                model = model_ex
                # При желании можно задать n_estimators из self.trees_count, но для чистоты
                # walk-forward оставим обучение до конца (или фиксированное число деревьев).
                # Если нужен early_stopping по числу деревьев - можно добавить.
                model.fit(X_train_scaled, y_h)
                models_temp.append(model)

            # === Предсказания ===
            # На тренировочных данных
            train_preds_list = []
            for model in models_temp:
                train_preds_list.append(model.predict(X_train_scaled))
            train_preds = np.column_stack(train_preds_list)  # (train_windows, horizon)

            # На валидационном окне
            val_preds_list = []
            for model in models_temp:
                val_preds_list.append(model.predict(X_val_scaled))
            val_preds = np.array(val_preds_list).flatten()  # (horizon,)

            # === Обратное масштабирование для расчёта метрик в исходных единицах ===
            y_train_inv = scaler_y_local.inverse_transform(y_train_scaled)
            train_preds_inv = scaler_y_local.inverse_transform(train_preds)

            y_val_true_inv = y_val_true  # уже в исходных единицах
            val_preds_inv = scaler_y_local.inverse_transform(val_preds.reshape(1, -1)).flatten()

            # === Метрики ===
            # Усредняем MSE по всем горизонтам
            train_error = mean_squared_error(y_train_inv.flatten(), train_preds_inv.flatten())
            val_error = mean_squared_error(y_val_true_inv, val_preds_inv)
            train_r2 = r2_score(y_train_inv.flatten(), train_preds_inv.flatten())
            val_r2 = r2_score(y_val_true_inv, val_preds_inv)

            all_train_errors.append(train_error)
            all_val_errors.append(val_error)
            all_train_r2.append(train_r2)
            all_val_r2.append(val_r2)

            # === Прогресс-бар ===
            percent = (iter_num / total_iterations) * 100
            filled = int(percent / 2)
            bar = '█' * filled + '░' * (50 - filled)
            print(
                f'\rWalk-Forward: |{bar}| {percent:.1f}% - Итерация {iter_num}/{total_iterations} - Val MSE: {val_error:.6f} - need time: {(time.time()-start_it)*(total_iterations-iter_num)}',
                end='', flush=True)

        print("\nWalk-Forward валидация завершена!")
        print(f"Средняя ошибка валидации (MSE): {np.mean(all_val_errors):.6f} +/- {np.std(all_val_errors):.6f}")
        print(f"Средний R² валидации: {np.mean(all_val_r2):.4f} +/- {np.std(all_val_r2):.4f}")
        print(f"Максимальная ошибка валидации: {np.max(all_val_errors):.6f}")
        print(f"Минимальная ошибка валидации: {np.min(all_val_errors):.6f}")
        if show_results:
            self.V.forward_validation_errors(all_val_errors, all_val_r2)
        # Если не нужно финальное обучение после walk-forward, можно завершить
        # self.is_fitted = True
        return

    def fit(self, data, feature_list, input_bars, horizon, trees_count, show_results=False, feature_func=None, target_func=None):
        self.input_bars = input_bars
        self.horizon = horizon
        self.trees_count = trees_count
        self.meta = {
            "input_bars": self.input_bars,
            "horizon": self.horizon,
            "trees_count": self.trees_count
        }
        x, y = self._DataSplitting(data, input_bars, horizon, True)
        XX = []
        YY = []

        lx = len(x)
        self.feature_list = feature_list
        if len(x) != len(y): raise "pizdec"
        start_ = time.time()
        for i in range(len(x)):
            startt = time.time()
            if feature_func == None:
                window_features = self._prepare_single_window_features(x[i], feature_list)
            else:
                window_features = feature_func(x[i])

            if target_func == None:
                window_targets = self._prepare_single_window_target(y[i])
            else:
                window_targets = target_func(y[i])

            #print(len(window_features))
            XX.append(window_features)
            YY.append(window_targets)

            percent = (float(i) / lx) * 100
            filled_chars = int((i / lx) * 50)
            bar = '█' * filled_chars + '░' * (50 - filled_chars)
            time_per = time.time() - startt
            if percent < 10:
                print(f'\rПодготовка данных: |{bar}|   {percent:.2f}%  Осталось {time_per*(lx-i):.2f} секунд', end='', flush=True)
            elif percent >= 10 and percent < 100:
                print(f'\rПодготовка данных: |{bar}|  {percent:.2f}%   Осталось {time_per*(lx-i):.2f} секунд', end='', flush=True)
            else:
                print(f'\rПодготовка данных: |{bar}| {percent:.2f}%    Осталось {time_per*(lx-i):.2f} секунд', end='', flush=True)

        print()
        print("time spended: ", time.time() - start_)
        x = np.array(XX)
        y = np.array(YY)

        print(x[52])
        print(y[52])

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=42)
        X_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        Y_scaled = self.scaler_y.fit_transform(y_train)
        Y_test_scaled = self.scaler_y.transform(y_test)
        self.X_shape = x.shape[1]

        self.train_errors = []
        self.val_errors = []
        self.train_r2 = []
        self.val_r2 = []

        self.best_val_error = float('inf')
        self.best_r2 = -float('inf')
        self.patience_counter = 0
        self.patience = 3

        trees = 1
        start = time.time()
        try:
            for i in range(trees, trees_count+1):
                print(f'{i} trees')
                t_error = 0
                v_error = 0
                t_r2 = 0
                v_r2 = 0
                if isinstance(horizon, int):
                    horizon_list = list(range(horizon))
                else:
                    horizon_list = horizon

                if len(Y_scaled.shape) == 2 and Y_scaled.shape[1] > 1:
                    for h_idx, h in enumerate(horizon_list):
                        if h_idx >= Y_scaled.shape[1]:
                            print(f"Предупреждение: горизонт {h} выходит за пределы y, пропускаем")
                            continue

                        y_h = Y_scaled.iloc[:, h_idx] if hasattr(Y_scaled, 'iloc') else Y_scaled[:, h_idx]

                        valid_mask = ~pd.isna(y_h) if hasattr(y_h, 'isna') else ~np.isnan(y_h)
                        X_h = X_scaled[valid_mask]
                        y_h_clean = y_h[valid_mask]

                        if i != 1:
                            self.models[h_idx].set_params(n_estimators=i)
                            self.models[h_idx].fit(X_h, y_h_clean)
                        else:
                            model = self.base_model.__class__(**self.base_model.get_params())
                            model.n_estimators = i
                            model.fit(X_h, y_h_clean)
                            self.models.append(model)

                        y_h_v = Y_test_scaled.iloc[:, h_idx] if hasattr(Y_test_scaled, 'iloc') else Y_test_scaled[:, h_idx]

                        valid_mask = ~pd.isna(y_h_v) if hasattr(y_h_v, 'isna') else ~np.isnan(y_h_v)
                        X_h_v = X_test_scaled[valid_mask]
                        y_h_v_clean = y_h_v[valid_mask]
                        if i != 1:
                            t_error += mean_squared_error(y_h_clean, self.models[h_idx].predict(X_h))
                            v_error += mean_squared_error(y_h_v_clean, self.models[h_idx].predict(X_h_v))
                            t_r2 += r2_score(y_h_clean, self.models[h_idx].predict(X_h))
                            v_r2 += r2_score(y_h_v_clean, self.models[h_idx].predict(X_h_v))
                        else:
                            t_error += mean_squared_error(y_h_clean, model.predict(X_h))
                            v_error += mean_squared_error(y_h_v_clean, model.predict(X_h_v))
                            t_r2 += r2_score(y_h_clean, model.predict(X_h))
                            v_r2 += r2_score(y_h_v_clean, model.predict(X_h_v))


                var_test_error = float(t_error)/horizon
                var_val_error = float(v_error)/horizon
                var_test_r2 = float(t_r2)/horizon
                var_val_r2 = float(v_r2)/horizon

                if self.early_stopping:
                    if len(self.val_errors) > 0:
                        current_min = min(self.val_errors)
                        best_so_far = min(self.best_val_error, current_min)

                        no_improvement_count = len(self.val_errors) - self.val_errors.index(best_so_far) - 1

                        if no_improvement_count >= self.patience:
                            print(f'Early stopping at {i} trees (no improvement for {self.patience} steps)')
                            if show_results:
                                self.V.show_errors(self.train_errors, self.val_errors,
                                                   self.train_r2, self.val_r2)
                            self.is_fitted = True
                            return

                self.train_errors.append(var_test_error)
                self.val_errors.append(var_val_error)
                self.train_r2.append(var_test_r2)
                self.val_r2.append(var_val_r2)
                print('Train error:      ', var_test_error)
                print('Validation error: ', var_val_error)
                print('Train r2:         ', var_test_r2)
                print('Validation r2:    ', var_val_r2)
                print(f"Затрачено времени: {time.time() - start} сек")

        except KeyboardInterrupt:
            print("\nОбучение прервано по Ctrl+C!")

        if show_results:
            self.V.show_errors(self.train_errors, self.val_errors, self.train_r2, self.val_r2)

        print('model is trained')
        self.is_fitted = True

        return self.train_errors, self.val_errors

    def forecast(self, latest_data, feature_func=None, show=False):
        if feature_func == None:
            X = self._prepare_single_window_features(latest_data, self.feature_list)
        else:
            X = feature_func(latest_data)

        print(len(latest_data))
        print(len(X))

        if not self.is_fitted and not self.onnx_load:
            raise ValueError("Модель еще не обучена!")

        if self.onnx_load:
            predictions = []

            X_float32 = X.astype(np.float32)
            if len(X_float32.shape) == 1:
                X_float32 = X_float32.reshape(1, -1)

            X_float32 = self.scaler.transform(X_float32)

            for i, session in enumerate(self.loaded_models):
                input_name = session.get_inputs()[0].name

                pred = session.run(None, {input_name: X_float32})
                predictions.append(float(f'{pred[0][0][0]:.7f}'))

            pred_array = np.array(predictions)

            # ИСПРАВЛЕНИЕ ЗДЕСЬ:
            if pred_array.ndim == 1:
                # Для одного сэмпла с 30 таргетами: (30,) -> (1, 30)
                pred_array = pred_array.reshape(1, -1)
            elif pred_array.ndim == 2:
                if pred_array.shape[0] == 1 and pred_array.shape[1] == 30:
                    # Уже правильно (1, 30) - ничего не делаем
                    pass
                elif pred_array.shape[0] == 30 and pred_array.shape[1] == 1:
                    # Неправильная форма (30, 1) -> (1, 30)
                    pred_array = pred_array.T
                elif pred_array.shape[0] > 1 and pred_array.shape[1] == 30:
                    # Несколько моделей: (n_models, 30) - берем первую или усредняем
                    pred_array = pred_array[0:1, :]  # Берем первую модель
            predictions = self.scaler_y.inverse_transform(pred_array).flatten()

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
                rez = np.concatenate([np.array(latest_data['TR'].values[1:]), np.array(predictions)])
                dates = pd.date_range(start='2024-01-01', periods=len(rez), freq='D')
                rez = pd.DataFrame(rez, index=dates, columns=['value'])
                self.V.show_vol(rez, len(predictions))
            return np.array(predictions)
        else:
            X = X.astype(np.float32)
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            X_scaled = self.scaler.transform(X)

            predictions = [] #jj
            for model in self.models:
                pred = model.predict(X_scaled)
                if len(pred.shape) > 0 and pred.shape[0] > 1:
                    predictions.append(pred)
                else:
                    predictions.append(pred[0] if len(pred.shape) > 0 else pred)

            pred_array = np.array(predictions)

            # ИСПРАВЛЕНИЕ ЗДЕСЬ:
            if pred_array.ndim == 1:
                # Для одного сэмпла с 30 таргетами: (30,) -> (1, 30)
                pred_array = pred_array.reshape(1, -1)
            elif pred_array.ndim == 2:
                if pred_array.shape[0] == 1 and pred_array.shape[1] == 30:
                    # Уже правильно (1, 30) - ничего не делаем
                    pass
                elif pred_array.shape[0] == 30 and pred_array.shape[1] == 1:
                    # Неправильная форма (30, 1) -> (1, 30)
                    pred_array = pred_array.T
                elif pred_array.shape[0] > 1 and pred_array.shape[1] == 30:
                    # Несколько моделей: (n_models, 30) - берем первую или усредняем
                    pred_array = pred_array[0:1, :]  # Берем первую модель
            predictions = self.scaler_y.inverse_transform(pred_array).flatten()

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
                rez = np.concatenate([np.array(latest_data['TR'].values[1:]), np.array(predictions)])
                dates = pd.date_range(start='2024-01-01', periods=len(rez), freq='D')
                rez = pd.DataFrame(rez, index=dates, columns=['value'])
                self.V.show_vol(rez, len(predictions))
            return np.array(predictions)


    def show_train_results(self):
        self.V.show_errors(self.train_errors, self.val_errors, self.train_r2, self.val_r2)


    def save_mql5(self, name):
        scaler_data = {
            "mean": self.scaler.mean_.tolist() if self.scaler.mean_ is not None else [],
            "std": self.scaler.scale_.tolist() if self.scaler.scale_ is not None else [],
            # scale_ = стандартное отклонение
            "var": self.scaler.var_.tolist() if self.scaler.var_ is not None else []
        }
        mean_str = ','.join(str(x) for x in scaler_data['mean'])
        std_str = ','.join(str(x) for x in scaler_data['std'])

        scaler_data_y = {
            "mean": self.scaler_y.mean_.tolist() if self.scaler_y.mean_ is not None else [],
            "std": self.scaler_y.scale_.tolist() if self.scaler_y.scale_ is not None else [],
            # scale_ = стандартное отклонение
            "var": self.scaler_y.var_.tolist() if self.scaler_y.var_ is not None else []
        }
        mean_str_y = ','.join(str(x) for x in scaler_data_y['mean'])
        std_str_y = ','.join(str(x) for x in scaler_data_y['std'])


        os.makedirs(name, exist_ok=True)
        print(f"Директория '{name}' создана или уже существует")

        # 2. Создаём папку для ONNX файлов
        onnx_dir = os.path.join(name, f"{name}_onnx")
        os.makedirs(onnx_dir, exist_ok=True)

        # 3. Создаём основной файл исходного кода MQL5 (.mq5)
        mq5_file_path = os.path.join(name, f"{name}.mq5")  # Изменено с .mql5 на .mq5
        with open(mq5_file_path, "w", encoding="utf-8") as f:
            f.write("//+------------------------------------------------------------------+\n")
            f.write(f"//|                                                  {name}.mq5 |\n")
            f.write("//|                                                  Made by DQuant. |\n")
            f.write("//|                                             https://dquant.space |\n")
            f.write("//+------------------------------------------------------------------+\n")
            f.write("#property indicator_buffers 2\n")
            f.write("#property indicator_plots   2\n")
            f.write("#property indicator_color1  clrGreen\n")
            f.write("#property indicator_width1  3\n")
            f.write("#property indicator_type1   DRAW_HISTOGRAM\n")
            f.write("#property indicator_color2  clrRed\n")
            f.write("#property indicator_width2  3\n")
            f.write("#property indicator_type2   DRAW_HISTOGRAM\n\n")

            f.write("//--- input parameters\n")
            f.write(f"int bars_to = {len(self.models)};\n")
            f.write(f"double forecast[{len(self.models)}];\n")
            f.write("datetime dt = NULL;\n\n")
            f.write(f"double mean_[] = {{{mean_str}}};\n\n")
            f.write(f"double std_[] = {{{std_str}}};\n\n")

            f.write(f"double mean_y[] = {{{mean_str_y}}};\n\n")
            f.write(f"double std_y[] = {{{std_str_y}}};\n\n")

            f.write("//--- indicator buffers\n")
            f.write("double past_vol[];\n")
            f.write("double future_vol[];\n\n")

            # Resource declarations
            for i in range(len(self.models)):
                f.write(f'#resource "\\\\{name}_onnx\\\\{name}_{i}.onnx" as uchar ExtModelData{i}[]\n')
            f.write("\n")

            # Model handles
            for i in range(len(self.models)):
                f.write(f'long model_handle{i} = INVALID_HANDLE;\n')
            f.write("\n")

            # OnInit function
            f.write('//+------------------------------------------------------------------+\n')
            f.write('//| Custom indicator initialization function                         |\n')
            f.write('//+------------------------------------------------------------------+\n')
            f.write('int OnInit()\n')
            f.write('{\n')
            f.write('   SetIndexBuffer(0, past_vol, INDICATOR_DATA);\n')
            f.write('   SetIndexBuffer(1, future_vol, INDICATOR_DATA);\n')
            f.write('   ArraySetAsSeries(past_vol, true);\n')
            f.write('   ArraySetAsSeries(future_vol, true);\n\n')

            for i in range(len(self.models)):
                f.write(f'   model_handle{i} = OnnxCreateFromBuffer(ExtModelData{i}, ONNX_USE_CPU_ONLY);\n')
                f.write(f'   if (model_handle{i} == INVALID_HANDLE)\n')
                f.write('   {\n')
                f.write(f'      Print("Ошибка загрузки ONNX-модели {i}: ", GetLastError());\n')
                f.write('      return INIT_FAILED;\n')
                f.write('   }\n')
                # f.write(f'   Print("ONNX-модель {i} успешно загружена");\n\n')

            f.write('   return(INIT_SUCCEEDED);\n')
            f.write('}\n\n')

            # OnCalculate function
            f.write('//+------------------------------------------------------------------+\n')
            f.write('//| Custom indicator iteration function                              |\n')
            f.write('//+------------------------------------------------------------------+\n')
            f.write('int OnCalculate(const int rates_total,\n')
            f.write('                const int prev_calculated,\n')
            f.write('                const datetime &time[],\n')
            f.write('                const double &open[],\n')
            f.write('                const double &high[],\n')
            f.write('                const double &low[],\n')
            f.write('                const double &close[],\n')
            f.write('                const long &tick_volume[],\n')
            f.write('                const long &volume[],\n')
            f.write('                const int &spread[])\n')
            f.write('{\n')
            f.write('   ArraySetAsSeries(high, true);\n')
            f.write('   ArraySetAsSeries(low, true);\n')
            f.write('   ArraySetAsSeries(close, true);\n')
            f.write('   ArraySetAsSeries(open, true);\n\n')

            f.write('   int counted_bars = prev_calculated;\n')
            f.write('   int limit;\n\n')

            f.write('   if(counted_bars < 0) return(-1);\n')
            f.write('   if(counted_bars > 0) counted_bars--;\n\n')

            f.write('   limit = MathMin(rates_total - 1, rates_total - counted_bars + bars_to);\n\n')

            f.write('   for(int i=limit; i >= bars_to; i--)\n')
            f.write('   {\n')
            f.write('         int j = i - bars_to;\n')
            f.write('         future_vol[i] = 0;\n')
            f.write('         \n')
            f.write('         if (i == limit)\n')
            f.write('         {\n')
            f.write('            past_vol[i] = MathAbs(high[j] - low[j]) / close[j];\n')
            f.write('         }\n')
            f.write('         else\n')
            f.write('         {\n')
            f.write('            double hl = MathAbs(high[j] - low[j]) / close[j];\n')
            f.write('            double hc = MathAbs(high[j] - close[j+1]) / close[j];\n')
            f.write('            double lc = MathAbs(low[j] - close[j+1]) / close[j];\n')
            f.write('            double mid = MathMax(hl, hc);\n')
            f.write('            past_vol[i] = MathMax(mid, lc);\n')
            f.write('            future_vol[i] = 0.0;\n')
            f.write('         }\n')
            f.write('   }\n\n')
            f.write('   if (!dt || dt != iTime(_Symbol, _Period, 0)){\n')
            f.write('       GetForecast();\n')
            f.write('   }\n')
            f.write('   return(rates_total);\n')
            f.write('}\n')

            # OnDeinit function
            f.write('\n//+------------------------------------------------------------------+\n')
            f.write('//| Custom indicator deinitialization function                       |\n')
            f.write('//+------------------------------------------------------------------+\n')
            f.write('void OnDeinit(const int reason)\n')
            f.write('{\n')
            for i in range(len(self.models)):
                f.write(f'   if(model_handle{i} != INVALID_HANDLE)\n')
                f.write(f'      OnnxRelease(model_handle{i});\n')
            f.write('}\n\n')

            # PrepareFeaturesForModel function
            f.write('//+------------------------------------------------------------------+\n')
            f.write('//| Prepare features in the same order as Python                     |\n')
            f.write('//| Order: TR, return, abs_return, gap, body, shadow, close_position |\n')
            f.write('//+------------------------------------------------------------------+\n')
            f.write(
                f'bool PrepareFeaturesForModel(double &feature_vector[], int window_size = {self.input_bars}, double epsilon = 1e-10)\n')
            f.write('{\n')
            f.write('    MqlRates rates_array[];\n')
            f.write('    \n')
            f.write('    // Copy rates - need window_size + 1 bars for shift operations\n')
            f.write('    int copied = CopyRates(Symbol(), Period(), 1, window_size + 1, rates_array);\n')
            f.write('    \n')
            f.write('    if(copied != window_size + 1) \n')
            f.write('    {\n')
            f.write('        Print("Error getting data. Got: ", copied, " Expected: ", window_size + 1);\n')
            f.write('        return false;\n')
            f.write('    }\n')
            f.write('    \n')
            f.write(f'    // Calculate total features: {len(self.roll_features)} features per bar\n')
            f.write(f'    int features_per_bar = {len(self.roll_features)};\n')
            f.write(f'    int total_features = window_size * features_per_bar + {len(self.single_features)};\n')
            f.write('    ArrayResize(feature_vector, total_features);\n')
            f.write('    \n')
            f.write('    // Arrays for each feature\n')
            f.write('   MqlDateTime timedt;\n')
            for i in self.roll_features:
                f.write(f'    double {i}[];\n')
            for i in self.single_features:
                f.write(f'    double {i};\n')
            f.write('    \n')
            for i in self.roll_features:
                f.write(f'    ArrayResize({i}, window_size);\n')
            f.write('    \n')
            f.write('    // Calculate features for each bar (from oldest to newest)\n')
            f.write('    for(int i = 0; i < window_size; i++) \n')
            f.write('    {\n')
            f.write('        // Current bar\n')
            f.write('        MqlRates current = rates_array[i];\n')
            f.write('        \n')
            f.write('        // Previous bar (for shift operations) - rates_array[0] is oldest\n')
            f.write('        MqlRates prev = (i > 0) ? rates_array[i - 1] : current;\n')
            f.write('        \n')
            f.write('        double close_price = current.close;\n')
            f.write('        double divisor = close_price + epsilon;\n')
            f.write('        \n')
            for i in self.roll_features:
                if i == 'high_low':
                    f.write('        // high_low\n')
                    f.write('        high_low[i] = (current.high - current.low) / divisor;\n')
                    f.write('        \n')
                elif i == 'TR':
                    f.write('        // TR (True Range)\n')
                    f.write('        {\n')
                    f.write('            double high_low_tr = (current.high - current.low) / divisor;\n')
                    f.write('            double hc = MathAbs(current.high - prev.close) / divisor;\n')
                    f.write('            double lc = MathAbs(current.low - prev.close) / divisor;\n')
                    f.write('            TR[i] = MathMax(high_low_tr, MathMax(hc, lc));\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i == 'parkinson':
                    f.write('        // parkinson\n')
                    f.write('        {\n')
                    f.write(
                        '            parkinson[i] = MathSqrt((1.0/(4.0*M_LN2))*MathPow(MathLog(current.high/current.low), 2));\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i == 'garman_klass':
                    f.write('        // garman_klass\n')
                    f.write('        {\n')
                    f.write(
                        '            if(current.high > 0 && current.low > 0 && current.open > 0 && current.close > 0)\n')
                    f.write('            {\n')
                    f.write('                double hl = MathPow(MathLog(current.high / current.low), 2);\n')
                    f.write('                double co = MathPow(MathLog(current.close / current.open), 2);\n')
                    f.write('                garman_klass[i] = MathSqrt(0.5 * hl - (2.0*M_LN2-1.0)*co);\n')
                    f.write('            }\n')
                    f.write('            else\n')
                    f.write('            {\n')
                    f.write('                garman_klass[i] = 0.0;\n')
                    f.write('            }\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i == 'rogers_satchell':
                    f.write('        // rogers_satchell\n')
                    f.write('        {\n')
                    f.write(
                        '            if(current.high > 0 && current.low > 0 && current.open > 0 && current.close > 0)\n')
                    f.write('            {\n')
                    f.write('                double h_o = MathLog(current.high / current.open);\n')
                    f.write('                double l_o = MathLog(current.low / current.open);\n')
                    f.write('                double c_o = MathLog(current.close / current.open);\n')
                    f.write('                rogers_satchell[i] = MathSqrt(h_o * (h_o - c_o) + l_o * (l_o - c_o));\n')
                    f.write('            }\n')
                    f.write('            else\n')
                    f.write('            {\n')
                    f.write('                rogers_satchell[i] = 0.0;\n')
                    f.write('            }\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i == 'returns':
                    f.write('        // returns (log returns)\n')
                    f.write('        {\n')
                    f.write('            if(i > 0 && prev.close > 0 && current.close > 0) \n')
                    f.write('            {\n')
                    f.write('                returns[i] = MathLog(current.close / prev.close);\n')
                    f.write('            } \n')
                    f.write('            else \n')
                    f.write('            {\n')
                    f.write('                returns[i] = 0.0;\n')
                    f.write('            }\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i == 'abs_returns':
                    f.write('        // abs_returns\n')
                    f.write('        {\n')
                    f.write('            if(i > 0 && prev.close > 0 && current.close > 0) \n')
                    f.write('            {\n')
                    f.write('                abs_returns[i] = MathAbs(MathLog(current.close / prev.close));\n')
                    f.write('            } \n')
                    f.write('            else \n')
                    f.write('            {\n')
                    f.write('                abs_returns[i] = 0.0;\n')
                    f.write('            }\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i == 'gap':
                    f.write('        // gap\n')
                    f.write('        {\n')
                    f.write('            if(i > 0 && prev.close > 0) \n')
                    f.write('            {\n')
                    f.write('                gap[i] = (current.open - prev.close) / divisor;\n')
                    f.write('            } \n')
                    f.write('            else \n')
                    f.write('            {\n')
                    f.write('                gap[i] = 0.0;\n')
                    f.write('            }\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i == 'body':
                    f.write('        // body\n')
                    f.write('        body[i] = MathAbs(current.close - current.open) / divisor;\n')
                    f.write('        \n')
                elif i == 'shadow':
                    f.write('        // shadow\n')
                    f.write('        {\n')
                    f.write('            double max_open_close = MathMax(current.open, current.close);\n')
                    f.write('            double min_open_close = MathMin(current.open, current.close);\n')
                    f.write(
                        '            shadow[i] = ((current.high - max_open_close) + (min_open_close - current.low)) / divisor;\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i == 'close_position':
                    f.write('        // close_position\n')
                    f.write('        {\n')
                    f.write('            double high_low_diff = current.high - current.low;\n')
                    f.write('            if(high_low_diff > epsilon) \n')
                    f.write('            {\n')
                    f.write('                close_position[i] = (current.close - current.low) / high_low_diff;\n')
                    f.write('            } \n')
                    f.write('            else \n')
                    f.write('            {\n')
                    f.write('                close_position[i] = 0.5;\n')
                    f.write('            }\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i[:9] == 'roll_rsi_':
                    window = ''
                    for j in range(9, len(i)):
                        try:
                            int(i[j])
                        except ValueError:
                            break
                        else:
                            window += i[j]
                    window = int(window)
                    f.write('        // roll_rsi\n')
                    f.write('        {\n')
                    f.write(f'            if(i >= {window - 1})\n')
                    f.write('            {\n')
                    f.write('                double gain_sum = 0.0;\n')
                    f.write('                double loss_sum = 0.0;\n')
                    f.write('                \n')
                    f.write(f'                for(int j = i - {window - 1}; j <= i; j++)\n')
                    f.write('                {\n')
                    f.write('                    if(j > 0)\n')
                    f.write('                    {\n')
                    f.write('                        double delta = rates_array[j].close - rates_array[j-1].close;\n')
                    f.write('                        if(delta > 0)\n')
                    f.write('                            gain_sum += delta;\n')
                    f.write('                        else\n')
                    f.write('                            loss_sum += -delta;\n')
                    f.write('                    }\n')
                    f.write('                }\n')
                    f.write('                \n')
                    f.write(f'                double gain = gain_sum / {window}.0;\n')
                    f.write(f'                double loss = loss_sum / {window}.0;\n')
                    f.write('                double rs = gain / (loss + epsilon);\n')
                    f.write(f'                roll_rsi_{window}[i] = 100.0 - (100.0 / (1.0 + rs));\n')
                    f.write('            }\n')
                    f.write('            else\n')
                    f.write('            {\n')
                    f.write(f'                roll_rsi_{window}[i] = 0.0;\n')
                    f.write('            }\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i[:9] == 'roll_atr_':
                    window = ''
                    for j in range(9, len(i)):
                        try:
                            int(i[j])
                        except ValueError:
                            break
                        else:
                            window += i[j]
                    window = int(window)
                    f.write('        // roll_atr\n')
                    f.write('        {\n')
                    f.write(f'            if(i >= {window - 1})\n')
                    f.write('            {\n')
                    f.write('                double tr_sum = 0.0;\n')
                    f.write(f'                for(int j = i - {window - 1}; j <= i; j++)\n')
                    f.write('                {\n')
                    f.write('                    tr_sum += TR[j];\n')
                    f.write('                }\n')
                    f.write(
                        f'                roll_atr_{window}[i] = (tr_sum / {window}.0) / (rates_array[i].close + epsilon);\n')
                    f.write('            }\n')
                    f.write('            else\n')
                    f.write('            {\n')
                    f.write(f'                roll_atr_{window}[i] = 0.0;\n')
                    f.write('            }\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i[:8] == 'roll_bb_':
                    window = ''
                    for j in range(8, len(i)):
                        try:
                            int(i[j])
                        except ValueError:
                            break
                        else:
                            window += i[j]
                    window = int(window)
                    f.write('        // roll_bb\n')
                    f.write('        {\n')
                    f.write(f'            if(i >= {window - 1})\n')
                    f.write('            {\n')
                    f.write('                double sma_sum = 0.0;\n')
                    f.write(f'                for(int j = i - {window - 1}; j <= i; j++)\n')
                    f.write('                {\n')
                    f.write('                    sma_sum += rates_array[j].close;\n')
                    f.write('                }\n')
                    f.write(f'                double sma = sma_sum / {window}.0;\n')
                    f.write('                \n')
                    f.write('                double variance = 0.0;\n')
                    f.write(f'                for(int j = i - {window - 1}; j <= i; j++)\n')
                    f.write('                {\n')
                    f.write('                    variance += MathPow(rates_array[j].close - sma, 2);\n')
                    f.write('                }\n')
                    f.write(f'                double std = MathSqrt(variance / {window}.0);\n')
                    f.write(f'                roll_bb_{window}[i] = (4.0 * std) / (sma + epsilon);\n')
                    f.write('            }\n')
                    f.write('            else\n')
                    f.write('            {\n')
                    f.write(f'                roll_bb_{window}[i] = 0.0;\n')
                    f.write('            }\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i == 'roll_month':
                    f.write('        // roll_month\n')
                    f.write('        {\n')
                    f.write('            MqlDateTime timedt;\n')
                    f.write('            TimeToStruct(iTime(_Symbol, _Period, window_size - i), timedt);\n')
                    f.write('            roll_month[i] = timedt.mon;\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i == 'roll_day_of_month':
                    f.write('        // roll_day_of_month\n')
                    f.write('        {\n')
                    f.write('            MqlDateTime timedt;\n')
                    f.write('            TimeToStruct(iTime(_Symbol, _Period, window_size - i), timedt);\n')
                    f.write('            roll_day_of_month[i] = timedt.day;\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i == 'roll_day_of_week':
                    f.write('        // roll_day_of_week\n')
                    f.write('        {\n')
                    f.write('            MqlDateTime timedt;\n')
                    f.write('            TimeToStruct(iTime(_Symbol, _Period, window_size - i), timedt);\n')
                    f.write('            roll_day_of_week[i] = timedt.day_of_week;\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i == 'roll_hour':
                    f.write('        // roll_hour\n')
                    f.write('        {\n')
                    f.write('            MqlDateTime timedt;\n')
                    f.write('            TimeToStruct(iTime(_Symbol, _Period, window_size - i), timedt);\n')
                    f.write('            roll_hour[i] = timedt.hour;\n')
                    f.write('        }\n')
                    f.write('        \n')

            f.write('    }\n')
            f.write('    int last_idx = window_size - 1;\n')

            for i in self.single_features:
                if i == 'month':
                    f.write('    {\n')
                    f.write('        MqlDateTime timedt;\n')
                    f.write('        TimeToStruct(iTime(_Symbol, _Period, 1), timedt);\n')
                    f.write('        month = timedt.mon;\n')
                    f.write('    }\n')
                elif i == 'day_of_month':
                    f.write('    {\n')
                    f.write('        MqlDateTime timedt;\n')
                    f.write('        TimeToStruct(iTime(_Symbol, _Period, 1), timedt);\n')
                    f.write('        day_of_month = timedt.day;\n')
                    f.write('    }\n')
                elif i == 'day_of_week':
                    f.write('    {\n')
                    f.write('        MqlDateTime timedt;\n')
                    f.write('        TimeToStruct(iTime(_Symbol, _Period, 1), timedt);\n')
                    f.write('        day_of_week = timedt.day_of_week;\n')
                    f.write('    }\n')
                elif i == 'hour':
                    f.write('    {\n')
                    f.write('        MqlDateTime timedt;\n')
                    f.write('        TimeToStruct(iTime(_Symbol, _Period, 1), timedt);\n')
                    f.write('        hour = timedt.hour;\n')
                    f.write('    }\n')
                elif i[:4] == 'rsi_':
                    window = ''
                    for j in range(4, len(i)):
                        try:
                            int(i[j])
                        except ValueError:
                            break
                        else:
                            window += i[j]
                    window = int(window)
                    f.write('    {\n')
                    f.write('        double gain_sum = 0.0;\n')
                    f.write('        double loss_sum = 0.0;\n')
                    f.write('        \n')
                    f.write(f'        for(int idx = last_idx - {window} + 1; idx <= last_idx; idx++)\n')
                    f.write('        {\n')
                    f.write('            double delta = rates_array[idx].close - rates_array[idx-1].close;\n')
                    f.write('            if(delta > 0)\n')
                    f.write('                gain_sum += delta;\n')
                    f.write('            else\n')
                    f.write('                loss_sum += -delta;\n')
                    f.write('        }\n')
                    f.write('        \n')
                    f.write(f'        double gain = gain_sum / {window};\n')
                    f.write(f'        double loss = loss_sum / {window};\n')
                    f.write('        double rs = gain / (loss + epsilon);\n')
                    f.write(f'        {i} = 100.0 - (100.0 / (1.0 + rs));\n')
                    f.write('    }\n')
                    f.write('    \n')
                elif i[:4] == 'atr_':
                    window = ''
                    for j in range(4, len(i)):
                        try:
                            int(i[j])
                        except ValueError:
                            break
                        else:
                            window += i[j]
                    window = int(window)
                    f.write('    {\n')
                    f.write('        double tr_sum = 0.0;\n')
                    f.write(f'        for(int idx = last_idx - {window} + 1; idx <= last_idx; idx++)\n')
                    f.write('        {\n')
                    f.write('            MqlRates current = rates_array[idx];\n')
                    f.write('            MqlRates prev = rates_array[idx - 1];\n')
                    f.write('            double divisor = current.close + epsilon;\n')
                    f.write('            double high_low = (current.high - current.low) / divisor;\n')
                    f.write('            double hc = MathAbs(current.high - prev.close) / divisor;\n')
                    f.write('            double lc = MathAbs(current.low - prev.close) / divisor;\n')
                    f.write('            tr_sum += MathMax(high_low, MathMax(hc, lc));\n')
                    f.write('        }\n')
                    f.write(f'        {i} = (tr_sum / {window}) / (rates_array[last_idx].close + epsilon);\n')
                    f.write('    }\n')
                    f.write('    \n')
                elif i[:3] == 'bb_':
                    window = ''
                    for j in range(3, len(i)):
                        try:
                            int(i[j])
                        except ValueError:
                            break
                        else:
                            window += i[j]
                    window = int(window)
                    f.write('    {\n')
                    f.write('        double sma_sum = 0.0;\n')
                    f.write(f'        for(int idx = last_idx - {window} + 1; idx <= last_idx; idx++)\n')
                    f.write('        {\n')
                    f.write('            sma_sum += rates_array[idx].close;\n')
                    f.write('        }\n')
                    f.write(f'        double sma = sma_sum / {window};\n')
                    f.write('        double variance = 0.0;\n')
                    f.write(f'        for(int idx = last_idx - {window} + 1; idx <= last_idx; idx++)\n')
                    f.write('        {\n')
                    f.write('            variance += MathPow(rates_array[idx].close - sma, 2);\n')
                    f.write('        }\n')
                    f.write(f'        double std = MathSqrt(variance / {window});\n')
                    f.write(f'        {i} = (4.0 * std) / (sma + epsilon);\n')
                    f.write('    }\n')
                    f.write('    \n')

            f.write('    \n')
            f.write('    // Flatten features in the same order as Python\n')
            f.write('    int idx = 0;\n')
            f.write('    for(int i = 0; i < window_size; i++)\n')
            f.write('    {\n')
            for i in self.roll_features:
                f.write(f'        feature_vector[idx++] = {i}[i];           // {i}\n')
            f.write('    }\n')
            for i in self.single_features:
                f.write(f'    feature_vector[idx++] = {i};\n')
            f.write('    \n')
            f.write('    // Normalize all features\n')
            f.write('    for(int i = 0; i < total_features; i++)\n')
            f.write('    {\n')
            f.write('        if(std_[i] != 0)\n')
            f.write('            feature_vector[i] = (feature_vector[i] - mean_[i]) / std_[i];\n')
            f.write('        else\n')
            f.write('            feature_vector[i] = 0;\n')
            f.write('    }\n')
            f.write('    return true;\n')
            f.write('}\n\n')

            # GetForecast function
            f.write('//+------------------------------------------------------------------+\n')
            f.write('//| Get forecast from ONNX model                                     |\n')
            f.write('//+------------------------------------------------------------------+\n')
            f.write('void GetForecast()\n')
            f.write('{\n')
            f.write('   double input_features[];\n')
            f.write('   \n')
            f.write('   if(PrepareFeaturesForModel(input_features))\n')
            f.write('   {\n')
            f.write('         double output[1];\n')
            f.write('         ulong input_shape_xgb[2] = {1, ArraySize(input_features)};\n')
            f.write('         ulong output_shape[2] = {1, 1};\n')
            f.write('         \n')
            for i in range(len(self.models)):
                f.write(f'         if(!OnnxSetInputShape(model_handle{i}, 0, input_shape_xgb))\n')
                f.write('         {\n')
                f.write('            Print("OnnxSetInputShape failed, error: ", GetLastError());\n')
                f.write('            return;\n')
                f.write('         }\n')
                f.write(f'         if(!OnnxSetOutputShape(model_handle{i}, 0, output_shape))\n')
                f.write('         {\n')
                f.write('            Print("OnnxSetOutputShape failed, error: ", GetLastError());\n')
                f.write('            return;\n')
                f.write('         }\n')
                f.write(f'         if(OnnxRun(model_handle{i}, ONNX_USE_CPU_ONLY, input_features, output))\n')
                f.write('         {\n')
                f.write(f'           int target_index = {len(self.models) - 1 - i};\n')
                f.write('           double scaled_prediction = output[0];\n')
                f.write('           double original_prediction = scaled_prediction * std_y[target_index] + mean_y[target_index];\n')
                f.write(f'           future_vol[target_index] = original_prediction;\n')
                f.write(f'           past_vol[target_index] = 0;\n')
                f.write('         }\n')
            f.write('   }\n')
            f.write('}\n\n')

        print(f"Файл {mq5_file_path} успешно создан")

        # 4. Создаём файл проекта MQL5 (.mqproj)
        proj_file_path = os.path.join(name, f"{name}.mqproj")
        with open(proj_file_path, "w", encoding="utf-8-sig") as f:
            f.write('{\n')
            f.write('  "platform"    :"mt5",\n')
            f.write('  "program_type":"indicator",\n')
            f.write('  "copyright"   :"Copyright 2025, MetaQuotes Ltd.",\n')
            f.write('  "link"        :"https:\\/\\/www.mql5.com",\n')
            f.write('  "version"     :"1.00",\n')
            f.write('  "cpu_architecture" :"0",\n')
            f.write('  "optimize"    :"1",\n')
            f.write('  "fpzerocheck" :"1",\n')
            f.write('  "tester_no_cache":"0",\n')
            f.write('  "tester_everytick_calculate":"0",\n')
            f.write('  "unicode_character_set":"0",\n')
            f.write('  "static_libraries":"0",\n\n')

            f.write('  "indicator":\n')
            f.write('  {\n')
            f.write('    "window":"1",\n')
            f.write('    "applied_price":"1"\n')
            f.write('  },\n\n')

            f.write('  "files":\n')
            f.write('  [\n')
            f.write('    {\n')
            f.write('      "path":"' + name + '.mq5",\n')
            f.write('      "compile":true,\n')
            f.write('      "relative_to_project":true\n')
            f.write('    }\n')
            f.write('  ]\n')
            f.write('}\n')

        print(f"Файл {proj_file_path} успешно создан")



class VolClustGB(FichEn):
    def __init__(self, sett, early_stopping=True):
        self.models = []
        self.scaler = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X_shape = 0
        self.is_fitted = False
        self.onnx_load = False
        self.early_stopping = early_stopping
        self.V = Visualization('dark')
        self.default_sett = {
            'loss': 'squared_error',
            'learning_rate': 0.01,
            'n_estimators': 1,
            'max_depth': 3,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'subsample': 0.8,
            'random_state': 42,
            'warm_start': True
        }
        self.meta = {
            "model_type": "gb",
            "model_settings": self.default_sett
        }
        if sett == {}:
            self.base_model = GradientBoostingRegressor(**self.default_sett)
        else:
            self.base_model = GradientBoostingRegressor(**sett)

    def save(self, name, type_to_save='default'):
        if type_to_save == 'default':
            os.makedirs(name, exist_ok=True)
            initial_type = [('float_input', FloatTensorType([None, self.X_shape]))]

            file_path = os.path.join(name, f"{name}_features.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.feature_list, f, ensure_ascii=False, indent=2)

            self.meta = {
                "model_type": "gb",
                "model_settings": self.default_sett,
                "input_bars": self.input_bars,
                "horizon": self.horizon,
                "trees_count": self.trees_count,
            }
            file_path = os.path.join(name, f"{name}_model_settings.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.meta, f, ensure_ascii=False, indent=2)

            if hasattr(self, 'scaler'):
                scaler_path = os.path.join(name, f"{name}_scaler.pkl")
                joblib.dump(self.scaler, scaler_path)

            if hasattr(self, 'scaler_y'):
                scaler_path = os.path.join(name, f"{name}_scaler_y.pkl")
                joblib.dump(self.scaler_y, scaler_path)

            for i in range(len(self.models)):
                onx = convert_sklearn(self.models[i], initial_types=initial_type, target_opset=12)

                file_path = os.path.join(name, f"{name}_{i}.onnx")

                with open(file_path, "wb") as f:
                    f.write(onx.SerializeToString())
        elif type_to_save == 'mql5':
            self.save_mql5(name)
            onnx_dir = os.path.join(name, f"{name}_onnx")
            os.makedirs(onnx_dir, exist_ok=True)
            print(f"Директория для ONNX файлов создана: {onnx_dir}")

            # 4. Сохраняем модели и scaler в поддиректорию
            initial_type = [('float_input', FloatTensorType([None, self.X_shape]))]

            if hasattr(self, 'scaler') and self.scaler is not None:
                scaler_path = os.path.join(onnx_dir, f"{name}_scaler.pkl")
                joblib.dump(self.scaler, scaler_path)
                print(f"Scaler сохранён в {scaler_path}")

            if hasattr(self, 'scaler_y') and self.scaler_y is not None:
                scaler_path = os.path.join(onnx_dir, f"{name}_scaler_y.pkl")
                joblib.dump(self.scaler_y, scaler_path)
                print(f"Scalery сохранён в {scaler_path}")

            for i in range(len(self.models)):
                onx = convert_sklearn(self.models[i], initial_types=initial_type, target_opset=12)
                file_path = os.path.join(onnx_dir, f"{name}_{i}.onnx")
                with open(file_path, "wb") as f:
                    f.write(onx.SerializeToString())
                print(f"Модель {i} сохранена в {file_path}")

            print(f"Все операции в директории '{name}' успешно завершены!")


    def load(self, name):
        self.loaded_models = []

        if not os.path.exists(name):
            raise FileNotFoundError(f"Папка {name} не найдена")

        try:
            file_path = os.path.join(name, f"{name}_features.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                self.feature_list = json.load(f)
        except FileNotFoundError:
            print(f'Model {name} is not valid, file {name}_features.json is not found')
            return

        try:
            file_path = os.path.join(name, f"{name}_model_settings.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                self.meta = json.load(f)
        except FileNotFoundError:
            print(f'Model {name} is not valid, file {name}_model_settings.json is not found')
            return

        if self.meta['model_type'] != 'gb':
            raise ValueError(f"Wrong model type, expected gb and not a {self.meta['model_type']}")

        scaler_files = [f for f in os.listdir(name) if f.endswith('_scaler.pkl')]
        if scaler_files:
            scaler_path = os.path.join(name, scaler_files[0])
            self.scaler = joblib.load(scaler_path)

        scaler_files = [f for f in os.listdir(name) if f.endswith('_scaler_y.pkl')]
        if scaler_files:
            scaler_path = os.path.join(name, scaler_files[0])
            self.scaler_y = joblib.load(scaler_path)

        model_files = [f for f in os.listdir(name) if f.endswith('.onnx')]

        if not model_files:
            raise FileNotFoundError(f"В папке {name} не найдено .onnx файлов")

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


        for model_file in model_files:
            model_path = os.path.join(name, model_file)
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

            input_info = session.get_inputs()[0]
            self.loaded_models.append(session)

        self.onnx_load = True




class VolClustXGB(FichEn):
    def __init__(self, sett, early_stopping=True):
        self.models = []
        self.scaler = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X_shape = 0
        self.is_fitted = False
        self.onnx_load = False
        self.early_stopping = early_stopping
        self.V = Visualization('dark')
        self.default_sett = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'n_estimators': 1,
            'max_depth': 3,
            'min_child_weight': 5,
            'subsample': 0.8,
            'random_state': 42
        }
        self.meta = {
            "model_type": "xgb",
            "model_settings": self.default_sett
        }
        if sett == {}:
            self.base_model = xgboost.XGBRegressor(**self.default_sett)
        else:
            self.base_model = xgboost.XGBRegressor(**sett)


    def save(self, name, type_to_save='default'):
        if type_to_save == 'default':
            os.makedirs(name, exist_ok=True)
            initial_type = [('float_input', FloatTensorType([None, self.X_shape]))]

            file_path = os.path.join(name, f"{name}_features.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.feature_list, f, ensure_ascii=False, indent=2)

            self.meta = {
                "model_type": "xgb",
                "model_settings": self.default_sett,
                "input_bars": self.input_bars,
                "horizon": self.horizon,
                "trees_count": self.trees_count,
            }
            file_path = os.path.join(name, f"{name}_model_settings.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.meta, f, ensure_ascii=False, indent=2)

            if hasattr(self, 'scaler'):
                scaler_path = os.path.join(name, f"{name}_scaler.pkl")
                joblib.dump(self.scaler, scaler_path)

            if hasattr(self, 'scaler_y'):
                scaler_path = os.path.join(name, f"{name}_scaler_y.pkl")
                joblib.dump(self.scaler_y, scaler_path)

            for i in range(len(self.models)):
                onx = onnxmltools.convert_xgboost(self.models[i], initial_types=initial_type, target_opset=12)

                file_path = os.path.join(name, f"{name}_{i}.onnx")
                onnxmltools.utils.save_model(onx, file_path)
        elif type_to_save == 'mql5':
            self.save_mql5(name)
            onnx_dir = os.path.join(name, f"{name}_onnx")
            os.makedirs(onnx_dir, exist_ok=True)
            print(f"Директория для ONNX файлов создана: {onnx_dir}")

            # 4. Сохраняем модели и scaler в поддиректорию
            initial_type = [('float_input', FloatTensorType([None, self.X_shape]))]

            if hasattr(self, 'scaler') and self.scaler is not None:
                scaler_path = os.path.join(onnx_dir, f"{name}_scaler.pkl")
                joblib.dump(self.scaler, scaler_path)
                print(f"Scaler сохранён в {scaler_path}")

            if hasattr(self, 'scaler_y') and self.scaler_y is not None:
                scaler_path = os.path.join(onnx_dir, f"{name}_scaler_y.pkl")
                joblib.dump(self.scaler_y, scaler_path)
                print(f"Scalery сохранён в {scaler_path}")

            for i in range(len(self.models)):
                onx = onnxmltools.convert_xgboost(self.models[i], initial_types=initial_type, target_opset=9)
                model_path = os.path.join(onnx_dir, f"{name}_{i}.onnx")
                onnxmltools.utils.save_model(onx, model_path)
                print(f"Модель {i} сохранена в {model_path}")

            print(f"Все операции в директории '{name}' успешно завершены!")



    def load(self, name):
        self.loaded_models = []

        if not os.path.exists(name):
            raise FileNotFoundError(f"Папка {name} не найдена")

        try:
            file_path = os.path.join(name, f"{name}_features.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                self.feature_list = json.load(f)
        except FileNotFoundError:
            print(f'Model {name} is not valid, file {name}_features.json is not found')
            return

        try:
            file_path = os.path.join(name, f"{name}_model_settings.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                self.meta = json.load(f)
        except FileNotFoundError:
            print(f'Model {name} is not valid, file {name}_model_settings.json is not found')
            return

        if self.meta['model_type'] != 'xgb':
            raise ValueError(f"Wrong model type, expected xgb and not a {self.meta['model_type']}")

        scaler_files = [f for f in os.listdir(name) if f.endswith('_scaler.pkl')]
        if scaler_files:
            scaler_path = os.path.join(name, scaler_files[0])
            self.scaler = joblib.load(scaler_path)

        scaler_files = [f for f in os.listdir(name) if f.endswith('_scaler_y.pkl')]
        if scaler_files:
            scaler_path = os.path.join(name, scaler_files[0])
            self.scaler_y = joblib.load(scaler_path)

        model_files = [f for f in os.listdir(name) if f.endswith('.onnx')]

        if not model_files:
            raise FileNotFoundError(f"В папке {name} не найдено .onnx файлов")

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


        for model_file in model_files:
            model_path = os.path.join(name, model_file)
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

            input_info = session.get_inputs()[0]
            self.loaded_models.append(session)

        self.onnx_load = True


class VolClustLightGBM(FichEn):
    def __init__(self, sett, early_stopping=True):
        self.models = []
        self.scaler = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X_shape = 0
        self.is_fitted = False
        self.onnx_load = False
        self.early_stopping = early_stopping
        self.V = Visualization('dark')
        self.default_sett = {
            'objective': 'regression',
            'num_leaves': 7,
            'min_data_in_leaf': 3,
            'min_data_in_bin': 3,
            'learning_rate': 0.1,
            'verbosity': -1,
            'seed': 42
        }
        self.meta = {
            "model_type": "lgbm",
            "model_settings": self.default_sett
        }
        if sett == {}:
            self.base_model = lgb.LGBMRegressor(**self.default_sett)
        else:
            self.base_model = lgb.LGBMRegressor(**sett)


    def save(self, name, type_to_save='default'):
        if type_to_save == 'default':
            os.makedirs(name, exist_ok=True)
            initial_type = [('float_input', FloatTensorType([None, self.X_shape]))]

            file_path = os.path.join(name, f"{name}_features.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.feature_list, f, ensure_ascii=False, indent=2)

            self.meta = {
                "model_type": "lgbm",
                "model_settings": self.default_sett,
                "input_bars": self.input_bars,
                "horizon": self.horizon,
                "trees_count": self.trees_count,
            }
            file_path = os.path.join(name, f"{name}_model_settings.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.meta, f, ensure_ascii=False, indent=2)

            if hasattr(self, 'scaler'):
                scaler_path = os.path.join(name, f"{name}_scaler.pkl")
                joblib.dump(self.scaler, scaler_path)

            if hasattr(self, 'scaler_y'):
                scaler_path = os.path.join(name, f"{name}_scaler_y.pkl")
                joblib.dump(self.scaler_y, scaler_path)

            for i in range(len(self.models)):
                onx = onnxmltools.convert_lightgbm(self.models[i], initial_types=initial_type, zipmap=False,
                                                   target_opset=12)
                file_path = os.path.join(name, f"{name}_{i}.onnx")
                onnxmltools.utils.save_model(onx, file_path)
        elif type_to_save == 'mql5':
            self.save_mql5(name)
            onnx_dir = os.path.join(name, f"{name}_onnx")
            os.makedirs(onnx_dir, exist_ok=True)
            print(f"Директория для ONNX файлов создана: {onnx_dir}")

            # 4. Сохраняем модели и scaler в поддиректорию
            initial_type = [('float_input', FloatTensorType([None, self.X_shape]))]

            if hasattr(self, 'scaler') and self.scaler is not None:
                scaler_path = os.path.join(onnx_dir, f"{name}_scaler.pkl")
                joblib.dump(self.scaler, scaler_path)
                print(f"Scaler сохранён в {scaler_path}")

            if hasattr(self, 'scaler_y') and self.scaler_y is not None:
                scaler_path = os.path.join(onnx_dir, f"{name}_scaler_y.pkl")
                joblib.dump(self.scaler_y, scaler_path)
                print(f"Scalery сохранён в {scaler_path}")

            for i in range(len(self.models)):
                onx = onnxmltools.convert_lightgbm(self.models[i], initial_types=initial_type, zipmap=False,
                                                   target_opset=12)
                file_path = os.path.join(onnx_dir, f"{name}_{i}.onnx")
                onnxmltools.utils.save_model(onx, file_path)
                print(f"Модель {i} сохранена в {file_path}")

            print(f"Все операции в директории '{name}' успешно завершены!")


    def load(self, name):
        self.loaded_models = []

        if not os.path.exists(name):
            raise FileNotFoundError(f"Папка {name} не найдена")

        try:
            file_path = os.path.join(name, f"{name}_features.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                self.feature_list = json.load(f)
        except FileNotFoundError:
            print(f'Model {name} is not valid, file {name}_features.json is not found')
            return

        try:
            file_path = os.path.join(name, f"{name}_model_settings.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                self.meta = json.load(f)
        except FileNotFoundError:
            print(f'Model {name} is not valid, file {name}_model_settings.json is not found')
            return

        if self.meta['model_type'] != 'lgbm':
            raise ValueError(f"Wrong model type, expected lgbm and not a {self.meta['model_type']}")

        scaler_files = [f for f in os.listdir(name) if f.endswith('_scaler.pkl')]
        if scaler_files:
            scaler_path = os.path.join(name, scaler_files[0])
            self.scaler = joblib.load(scaler_path)

        scaler_files = [f for f in os.listdir(name) if f.endswith('_scaler_y.pkl')]
        if scaler_files:
            scaler_path = os.path.join(name, scaler_files[0])
            self.scaler_y = joblib.load(scaler_path)

        model_files = [f for f in os.listdir(name) if f.endswith('.onnx')]

        if not model_files:
            raise FileNotFoundError(f"В папке {name} не найдено .onnx файлов")

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


        for model_file in model_files:
            model_path = os.path.join(name, model_file)
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

            input_info = session.get_inputs()[0]
            self.loaded_models.append(session)

        self.onnx_load = True


"""
time spended:  65.44393944740295
[ 1.00901477e-02  6.06840734e-03  7.89991808e-03  2.40415813e-03
  1.30973943e-04  2.53802438e-03  7.55212328e-03  6.40525625e-01
  0.00000000e+00  2.81739978e-02  1.71405577e-02  7.20318643e-03
  2.66850655e-02  1.37758612e-05  2.63183893e-02  1.85560851e-03
  9.53905366e-01  0.00000000e+00  4.29736387e-02  2.55626153e-02
  2.36549673e-02  2.80319922e-02 -3.03892649e-04  2.81246929e-02
  1.48489457e-02  2.72801986e-01  0.00000000e+00  3.72218795e-02
  2.20329899e-02  1.21903154e-02  3.24872375e-02  1.21786687e-04
  3.31424959e-02  4.07938359e-03  1.05498777e-01  0.00000000e+00
  7.37316373e-02  4.32737643e-02  3.21723278e-02  5.74076879e-02
 -3.87169703e-04  5.87003299e-02  1.50313074e-02  1.78512223e-01
  0.00000000e+00  4.04910463e-02  2.44025016e-02  2.97286458e-02
  4.09296639e-03  5.20381594e-04  4.62173561e-03  3.58693107e-02
  5.82791094e-01  0.00000000e+00  4.58374744e-02  2.74720640e-02
  3.16518085e-02  1.28362178e-02  4.88255628e-04  1.34072113e-02
  3.24302631e-02  4.51602205e-01  0.00000000e+00  2.04805939e-02
  1.21755897e-02  1.55781241e-02  8.44683596e-03 -1.44448658e-04
  8.33816248e-03  1.21424314e-02  0.00000000e+00  0.00000000e+00
  2.98493845e-02  1.78514990e-02  2.09793890e-02  4.31824090e-03
  2.91972368e-05  4.35677518e-03  2.54926093e-02  3.56991113e-01
  0.00000000e+00  4.38209920e-02  2.67519393e-02  1.94066006e-02
  3.52507396e-02  9.13841081e-05  3.45452848e-02  9.27570718e-03
  8.67142706e-01  0.00000000e+00  2.23729157e-02  1.34131927e-02
  1.50097284e-02  9.27476481e-03 -4.87292988e-04  8.83061573e-03
  1.35423000e-02  4.21120863e-01  0.00000000e+00  1.49549846e-02
  8.95837170e-03  1.02428690e-02  3.70974367e-03  8.63321678e-05
  3.80296545e-03  1.11520191e-02  3.26991207e-01  0.00000000e+00
  4.26058712e-02  2.58258056e-02  1.75734110e-02  3.62848420e-02
  5.55038464e-04  3.50793990e-02  7.30531730e-03  8.35520105e-01
  0.00000000e+00  1.48660486e-02  8.95036835e-03  9.96263886e-03
  4.81100437e-03 -4.97779422e-05  4.84922797e-03  1.00168207e-02
  6.67079520e-01  0.00000000e+00  3.05811410e-02  1.81291235e-02
  1.35250294e-02  2.35798305e-02 -1.06096226e-04  2.37539365e-02
  6.82720443e-03  7.05013333e-02  0.00000000e+00  9.97877681e-02
  5.71429634e-02  9.93251430e-03  9.43035288e-02 -2.15592232e-04
  9.86776488e-02  1.11011933e-03  3.53888733e-03  0.00000000e+00
  6.15493679e-02  3.69277996e-02  4.30480571e-02  2.30142933e-02
  3.82434298e-04  2.36635998e-02  3.78857681e-02  4.78877653e-01
  0.00000000e+00  4.07323827e-02  2.42627809e-02  3.03729349e-02
  1.75768508e-03 -1.89331202e-04  1.56989951e-03  3.91624832e-02
  2.94759703e-01  0.00000000e+00  3.89312610e-02  2.37273964e-02
  3.48261445e-02  2.14988488e-04  4.55482200e-05  1.69417160e-04
  3.87618438e-02  8.72129781e-01  0.00000000e+00  6.17359291e-01
  2.93932853e-01  1.08930483e-01  4.64730175e-01  4.39776373e-04
  5.92024459e-01  2.53348322e-02  3.59865231e-02  0.00000000e+00
  3.11147546e-01  2.11229224e-01  2.60957183e-01  1.12686222e-01
  8.45532816e-03  9.81137169e-02  2.13033829e-01  8.41486709e-01
  0.00000000e+00  9.61773087e-02  5.59225605e-02  4.42007214e-02
  6.75355700e-02  1.80191261e-03  7.16702268e-02  2.45070819e-02
  1.50546410e-01  0.00000000e+00  1.23761654e-01  7.29214118e-02
  9.68913447e-02  3.62457326e-02  1.29941148e-04  3.54667798e-02
  8.82948742e-02  3.34199130e-01  0.00000000e+00  1.62919592e-01
  9.78764618e-02  1.22212154e-01  7.26450577e-02 -1.41299781e-03
  7.39357846e-02  8.75708093e-02  5.42211975e-01  0.00000000e+00
  7.45248895e-02  4.52017387e-02  4.64108595e-02  4.12455257e-02
 -2.27759272e-03  4.26840965e-02  3.18407930e-02  6.25823123e-01
  0.00000000e+00  5.01099799e-02  3.03195942e-02  3.68273961e-02
  2.44820439e-03  2.83362114e-04  2.16184786e-03  4.79481321e-02
  6.44204942e-01  0.00000000e+00  1.76503480e-01  1.13815750e-01
  6.65450214e-02  1.67104324e-01  1.12708932e-03  1.52761574e-01
  2.37419066e-02  8.73218209e-01  0.00000000e+00  1.57850455e-01
  9.26514801e-02  1.13630263e-01  1.22445001e-03  7.43594102e-05
  1.14934127e-03  1.56701114e-01  3.40320281e-01  0.00000000e+00
  7.19979354e-02  4.34659656e-02  5.21177072e-02  2.21448508e-03
  1.25191284e-03  3.46885170e-03  6.85290837e-02  5.66441411e-01
  0.00000000e+00  9.19313837e-02  5.28749679e-02  4.98083731e-02
  5.90770239e-02  8.43358015e-05  6.09412847e-02  3.09900990e-02
  1.22037653e-02  0.00000000e+00  1.02696046e-01  6.47826521e-02
  3.54393880e-02  9.57832377e-02  1.74497187e-04  9.11645446e-02
  1.15315016e-02  9.58084134e-01  0.00000000e+00  5.61199048e-02
  3.44009941e-02  2.49703201e-02  4.84447086e-02  3.01829683e-03
  4.42716886e-02  1.18482162e-02  8.56546369e-01  0.00000000e+00
  5.32228122e-02  3.18107827e-02  3.69621168e-02  8.01155994e-03
  5.85698106e-04  8.62943647e-03  4.45933758e-02  4.05350998e-01
  0.00000000e+00  2.15145177e-02  1.30245089e-02  1.62742830e-02
  5.28121136e-03 -8.77263769e-04  6.14455405e-03  1.53699637e-02
  8.68350803e-01  0.00000000e+00  5.05633636e-02  2.96374579e-02
  2.35973846e-02  3.74133767e-02  4.55768528e-04  3.85778362e-02
  1.19855274e-02  9.46606724e-03  0.00000000e+00  5.63840947e-02
  3.33798150e-02  3.34520599e-02  3.58132408e-02 -4.07618484e-04
  3.60546411e-02  1.99614159e-02  3.55645082e-01  0.00000000e+00
  5.57884077e-02  3.26136331e-02  7.71543102e-03  5.26502133e-02
  5.79304149e-04  5.46401883e-02  1.14821941e-03  5.92355248e-03
  0.00000000e+00  9.54862414e-02  5.94218027e-02  4.00815346e-02
  8.22684486e-02  5.43653418e-04  7.84316689e-02  1.70545725e-02
  8.57720697e-01  0.00000000e+00  2.02454145e-02  1.21577151e-02
  1.43197749e-02  1.36810741e-03  1.18759225e-04  1.24841275e-03
  1.89970017e-02  4.94675292e-01  0.00000000e+00  6.20877978e-02
  3.84604802e-02  4.87031762e-02  2.57778032e-02 -2.00581227e-04
  2.56489734e-02  3.64388244e-02  9.85866990e-01  0.00000000e+00
  7.24692836e-02  4.32326577e-02  5.50895288e-02  2.78887560e-02
  0.00000000e+00  2.75034548e-02  4.49658288e-02  4.01572960e-01
  0.00000000e+00  4.89923950e-02  2.89772392e-02  3.65664667e-02
  8.90628383e-03  5.60189477e-04  9.50625226e-03  3.94861427e-02
  1.82037110e-01  0.00000000e+00  2.65697886e-02  1.61453720e-02
  1.38432176e-02  1.97258038e-02  7.27424183e-04  1.88050990e-02
  7.76468969e-03  9.37379581e-01  0.00000000e+00  2.46622989e-02
  1.47933610e-02  1.67767982e-02  1.11868608e-02 -7.34744665e-04
  1.05149231e-02  1.41473759e-02  4.48996965e-01  0.00000000e+00
  6.72313405e-02  4.17978961e-02  7.27511817e-03  6.83840155e-02
 -4.23498691e-04  6.65217268e-02  7.09613790e-04  1.00000000e+00
  0.00000000e+00  4.05808633e-02  2.40200436e-02  2.87843914e-02
  1.32013831e-02  2.59571539e-04  1.35484777e-02  2.70323856e-02
  1.36321339e-01  0.00000000e+00  2.76677447e-02  1.67989744e-02
  1.27991081e-02  2.17346261e-02  3.91139046e-04  2.11089920e-02
  6.55875268e-03  8.90965792e-01  0.00000000e+00  2.22339109e-02
  1.34302943e-02  1.93488183e-02  4.37399798e-03  5.29667003e-04
  4.91324488e-03  1.73206660e-02  7.57573661e-01  0.00000000e+00
  7.30232075e-02  4.27061128e-02  2.56072816e-02  6.16526462e-02
  2.51412998e-04  6.38442505e-02  9.17895698e-03  1.25699176e-01
  0.00000000e+00  1.98494355e-02  1.19230831e-02  1.40594483e-02
  9.34113613e-04  2.83895912e-04  1.21844594e-03  1.86309896e-02
  5.07988653e-01  0.00000000e+00  4.43071401e-02  2.66348271e-02
  3.08453405e-02  1.61980943e-02 -1.45620908e-04  1.62132316e-02
  2.80939085e-02  5.18062994e-01  0.00000000e+00  4.42411056e-02
  2.62008186e-02  3.37862051e-02  1.82479034e-02 -7.99863781e-04
  1.76155499e-02  2.58256919e-02  5.94497091e-01  0.00000000e+00
  2.40756419e-02  1.43885928e-02  1.82440682e-02  3.81350856e-04
 -2.56613507e-04  1.24810073e-04  2.39508318e-02  2.95056894e-01
  0.00000000e+00  4.44530014e-02  2.61539413e-02  2.34725191e-02
  2.97129371e-02  4.71806267e-04  3.06305774e-02  1.38224240e-02
  2.94891979e-02  0.00000000e+00  8.13491945e-02  5.08256128e-02
  3.51490041e-02  6.90291127e-02 -2.32655230e-04  6.69331464e-02
  1.44160481e-02  9.69519652e-01  0.00000000e+00  1.64667468e-02
  9.87202072e-03  1.14533989e-02  2.90152110e-03 -3.54366267e-05
  2.87029796e-03  1.35964488e-02  3.92401519e-01  0.00000000e+00
  2.48991200e-02  1.51168312e-02  7.20748225e-03  2.25009020e-02
 -5.36407562e-04  2.27860523e-02  2.11306769e-03  9.31981961e-01
  0.00000000e+00  1.57825262e-02  9.43347831e-03  8.63093904e-03
  9.44698796e-03  4.53071603e-04  9.94482320e-03  5.83770297e-03
  1.97150369e-01  0.00000000e+00  5.88186630e-02  3.45510576e-02
  2.67466667e-02  4.37078926e-02 -3.70718529e-04  4.43064339e-02
  1.45122291e-02  1.14765291e-01  0.00000000e+00  1.45500976e-02
  8.73308037e-03  1.03194872e-02  2.37642848e-04 -3.16019514e-04
  7.83484265e-05  1.44717492e-02  4.58307320e-01  0.00000000e+00
  3.90720635e-02  2.38379593e-02  1.46013508e-02  3.38498125e-02
 -1.24039168e-04  3.34073566e-02  5.66470690e-03  8.96944072e-01
  0.00000000e+00  5.52093720e-02  3.38163892e-02  2.62722462e-02
  4.29733472e-02  5.51653396e-04  4.15114252e-02  1.36979468e-02
  8.48703084e-01  0.00000000e+00  1.91329742e-02  1.12056766e-02
  7.58118553e-03  1.61780845e-02  5.90264833e-04  1.54576573e-02
  3.08505200e-03  8.33624529e-01  0.00000000e+00  1.58114158e-02
  9.48141355e-03  1.16968974e-02  2.51773978e-03 -5.52143041e-05
  2.56978724e-03  1.32416286e-02  4.03234229e-01  0.00000000e+00
  1.81235587e-02  1.09543091e-02  7.99943802e-03  1.44176361e-02
  2.64490251e-05  1.42877506e-02  3.83580801e-03  8.51084356e-01
  0.00000000e+00  1.49035498e-02  9.01786107e-03  0.00000000e+00
  1.49573560e-02 -5.74993747e-05  1.49035498e-02  0.00000000e+00
  1.00000000e+00  0.00000000e+00  1.07237191e-02  6.46880826e-03
  9.26255244e-03  1.46866263e-03  1.75372147e-04  1.29221253e-03
  9.43150654e-03  9.10789689e-01  0.00000000e+00  1.23360969e-01
  7.83966160e-02  3.67183038e-02  1.19841406e-01 -3.93352924e-05
  1.12978228e-01  1.03827418e-02  9.34866426e-01  0.00000000e+00
  1.04797471e-01  6.06890467e-02  8.08996063e-02  1.63219782e-02
 -3.88998460e-04  1.60669109e-02  8.87305601e-02  1.38085306e-01
  0.00000000e+00  4.39345720e-02  2.57854018e-02  2.96597192e-02
  2.35369104e-02  1.60506376e-03  2.16570141e-02  2.12386179e-02
  5.18074610e-01  0.00000000e+00]
[0.02178552 0.03785454 0.03499315 0.02283531 0.04798502 0.08585039
 0.023324   0.03473237 0.13709623 0.07662905 0.04318881 0.05538318
 0.05529575 0.05300378 0.03236753 0.04895759 0.0345107  0.03046795
 0.03755051 0.0754602 ]
"""