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
from sklearn.metrics import mean_squared_error
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

        for i in range(window_in, len(data) - window_out + 1):
            x_window = data.iloc[i - window_in:i]

            y_window = data.iloc[i:i + window_out]

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
                data[i] = last_datetime.dt.weekday
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

        feature_vector = []
        for i in range(len(data)):
            for feat in existing_features:
                feature_vector.append(data[feat].iloc[i])

        for i in single_feature_list_final:
            feature_vector.append(single_feature_dict[i])

        self.roll_features = existing_features
        self.single_features = single_feature_list_final

        return np.array(feature_vector)


    def _prepare_single_window_target(self, df_window: pd.DataFrame) -> np.ndarray:
        data = df_window.copy()

        required = ['open', 'high', 'low', 'close']

        for col in required:
            if col not in data.columns:
                raise ValueError(f"Колонка '{col}' не найдена")

        tr_values = []

        for i in range(len(data)):
            if i == 0:
                tr = (data['high'].iloc[i] - data['low'].iloc[i]) / data['close'].iloc[i]
            else:
                hl = data['high'].iloc[i] - data['low'].iloc[i]
                hc = abs(data['high'].iloc[i] - data['close'].iloc[i - 1])
                lc = abs(data['low'].iloc[i] - data['close'].iloc[i - 1])
                tr = max(hl/data['close'].iloc[i], hc/data['close'].iloc[i], lc/data['close'].iloc[i])

            tr_values.append(tr)

        return np.array(tr_values)


    def fit(self, data, feature_list, input_bars, horizon, trees_count, show_results=False, feature_func=None, target_func=None):
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
                window_targets = target_func(x[i])

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
                if isinstance(horizon, int):
                    horizon_list = list(range(horizon))
                else:
                    horizon_list = horizon

                if len(y_train.shape) == 2 and y_train.shape[1] > 1:
                    for h_idx, h in enumerate(horizon_list):
                        if h_idx >= y_train.shape[1]:
                            print(f"Предупреждение: горизонт {h} выходит за пределы y, пропускаем")
                            continue

                        y_h = y_train.iloc[:, h_idx] if hasattr(y_train, 'iloc') else y_train[:, h_idx]

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

                        y_h_v = y_test.iloc[:, h_idx] if hasattr(y_test, 'iloc') else y_test[:, h_idx]

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
            X = self._prepare_single_window_features(latest_data, self.feature_list)
        else:
            X = feature_func(latest_data)

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
            X = X.astype(np.float32)
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            X_scaled = self.scaler.transform(X)

            predictions = []
            for model in self.models:
                pred = model.predict(X_scaled)
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


    def save_mql5(self, name):
        scaler_data = {
            "mean": self.scaler.mean_.tolist() if self.scaler.mean_ is not None else [],
            "std": self.scaler.scale_.tolist() if self.scaler.scale_ is not None else [],
            # scale_ = стандартное отклонение
            "var": self.scaler.var_.tolist() if self.scaler.var_ is not None else []
        }
        mean_str = ','.join(str(x) for x in scaler_data['mean'])
        std_str = ','.join(str(x) for x in scaler_data['std'])
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
            f.write("//|                                                  Made by dquant. |\n")
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
                    f.write('        double high_low = (current.high - current.low) / divisor;\n')
                    f.write('        double hc = MathAbs(current.high - prev.close) / divisor;\n')
                    f.write('        double lc = MathAbs(current.low - prev.close) / divisor;\n')
                    f.write('        TR[i] = MathMax(high_low, MathMax(hc, lc));\n')
                    f.write('        \n')
                elif i == 'parkinson':
                    f.write('        // parkinson\n')
                    f.write(
                        '        parkinson[i] = MathSqrt((1.0/(4.0*M_LN2))*MathPow(MathLog(current.high/current.low), 2));\n')
                    f.write('        \n')
                elif i == 'garman_klass':
                    f.write('        // garman_klass\n')
                    f.write(
                        '        if(current.high > 0 && current.low > 0 && current.open > 0 && current.close > 0)\n')
                    f.write('        {\n')
                    f.write('            double hl = MathPow(MathLog(current.high / current.low), 2);\n')
                    f.write('            double co = MathPow(MathLog(current.close / current.open), 2);\n')
                    f.write('            garman_klass[i] = MathSqrt(0.5 * hl - (2.0*M_LN2-1.0)*co);\n')
                    f.write('        }\n')
                    f.write('        else\n')
                    f.write('        {\n')
                    f.write('            garman_klass[i] = 0.0;\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i == 'rogers_satchell':
                    f.write('        // rogers_satchell\n')
                    f.write(
                        '        if(current.high > 0 && current.low > 0 && current.open > 0 && current.close > 0)\n')
                    f.write('        {\n')
                    f.write('            double h_o = MathLog(current.high / current.open);\n')
                    f.write('            double l_o = MathLog(current.low / current.open);\n')
                    f.write('            double c_o = MathLog(current.close / current.open);\n')
                    f.write('            rogers_satchell[i] = MathSqrt(h_o * (h_o - c_o) + l_o * (l_o - c_o));\n')
                    f.write('        }\n')
                    f.write('        else\n')
                    f.write('        {\n')
                    f.write('            rogers_satchell[i] = 0.0;\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i == 'returns':
                    f.write('        // returns (log returns)\n')
                    f.write('        if(i > 0 && prev.close > 0 && current.close > 0) \n')
                    f.write('        {\n')
                    f.write('            returns[i] = MathLog(current.close / prev.close);\n')
                    f.write('        } \n')
                    f.write('        else \n')
                    f.write('        {\n')
                    f.write('            returns[i] = 0.0;\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i == 'abs_returns':
                    f.write('        // abs_returns\n')
                    f.write('        if(i > 0 && prev.close > 0 && current.close > 0) \n')
                    f.write('        {\n')
                    f.write('            abs_returns[i] = MathAbs(MathLog(current.close / prev.close));\n')
                    f.write('        } \n')
                    f.write('        else \n')
                    f.write('        {\n')
                    f.write('            abs_returns[i] = 0.0;\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i == 'gap':
                    f.write('        // gap\n')
                    f.write('        if(i > 0 && prev.close > 0) \n')
                    f.write('        {\n')
                    f.write('            gap[i] = (current.open - prev.close) / divisor;\n')
                    f.write('        } \n')
                    f.write('        else \n')
                    f.write('        {\n')
                    f.write('            gap[i] = 0.0;\n')
                    f.write('        }\n')
                    f.write('        \n')
                elif i == 'body':
                    f.write('        // body\n')
                    f.write('        body[i] = MathAbs(current.close - current.open) / divisor;\n')
                    f.write('        \n')
                elif i == 'shadow':
                    f.write('        // shadow\n')
                    f.write('        double max_open_close = MathMax(current.open, current.close);\n')
                    f.write('        double min_open_close = MathMin(current.open, current.close);\n')
                    f.write(
                        '        shadow[i] = ((current.high - max_open_close) + (min_open_close - current.low)) / divisor;\n')
                    f.write('        \n')
                elif i == 'close_position':
                    f.write('        // close_position\n')
                    f.write('        double high_low_diff = current.high - current.low;\n')
                    f.write('        if(high_low_diff > epsilon) \n')
                    f.write('        {\n')
                    f.write('            close_position[i] = (current.close - current.low) / high_low_diff;\n')
                    f.write('        } \n')
                    f.write('        else \n')
                    f.write('        {\n')
                    f.write('            close_position[i] = 0.5;\n')
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
                    f.write('        \n')
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
                elif i == 'roll_month':
                    f.write('   TimeToStruct(iTime(_Symbol, _Period, window_size-i), timedt);\n')
                    f.write('   roll_month[i] = timedt.mon;\n')
                elif i == 'roll_day_of_month':
                    f.write('   TimeToStruct(iTime(_Symbol, _Period, window_size-i), timedt);\n')
                    f.write('   roll_day_of_month[i] = timedt.day;\n')
                elif i == 'roll_day_of_week':
                    f.write('   TimeToStruct(iTime(_Symbol, _Period, window_size-i), timedt);\n')
                    f.write('   roll_day_of_week[i] = timedt.day_of_week;\n')
                elif i == 'roll_hour':
                    f.write('   TimeToStruct(iTime(_Symbol, _Period, window_size-i), timedt);\n')
                    f.write('   roll_hour[i] = timedt.hour;\n')
            f.write('    }\n')
            f.write('    int last_idx = window_size - 1;\n')
            for i in self.single_features:
                if i == 'month':
                    f.write('   TimeToStruct(iTime(_Symbol, _Period, 1), timedt);\n')
                    f.write('   month = timedt.mon;\n')
                elif i == 'day_of_month':
                    f.write('   TimeToStruct(iTime(_Symbol, _Period, 1), timedt);\n')
                    f.write('   day_of_month = timedt.day;\n')
                elif i == 'day_of_week':
                    f.write('   TimeToStruct(iTime(_Symbol, _Period, 1), timedt);\n')
                    f.write('   day_of_week = timedt.day_of_week;\n')
                elif i == 'hour':
                    f.write('   TimeToStruct(iTime(_Symbol, _Period, 1), timedt);\n')
                    f.write('   hour = timedt.hour;\n')
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
                    f.write('    \n')
                    f.write('    double gain_sum = 0.0;\n')
                    f.write('    double loss_sum = 0.0;\n')
                    f.write('    \n')
                    f.write(f'    for(int i = last_idx - {window} + 1; i <= last_idx; i++)\n')
                    f.write('    {\n')
                    f.write('        double delta = rates_array[i].close - rates_array[i-1].close;\n')
                    f.write('        \n')
                    f.write('        if(delta > 0)\n')
                    f.write('            gain_sum += delta;\n')
                    f.write('        else\n')
                    f.write('            loss_sum += -delta;\n')
                    f.write('    }\n')
                    f.write('    \n')
                    f.write(f'    double gain = gain_sum / {window};\n')
                    f.write(f'    double loss = loss_sum / {window};\n')
                    f.write('    \n')
                    f.write(f'    double rs = gain / (loss + epsilon);\n')
                    f.write(f'    {i} = 100.0 - (100.0 / (1.0 + rs));\n')
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
                    f.write('    double tr_sum = 0.0;\n')
                    f.write('    \n')
                    f.write(f'    for(int i = last_idx - {window} + 1; i <= last_idx; i++)\n')
                    f.write('    {\n')
                    f.write('        MqlRates current = rates_array[i];\n')
                    f.write('        MqlRates prev = rates_array[i - 1];\n')
                    f.write('        double divisor = current.close + epsilon;\n')
                    f.write('        double high_low = (current.high - current.low) / divisor;\n')
                    f.write('        double hc = MathAbs(current.high - prev.close) / divisor;\n')
                    f.write('        double lc = MathAbs(current.low - prev.close) / divisor;\n')
                    f.write('        tr_sum += MathMax(high_low, MathMax(hc, lc));\n')
                    f.write('    }\n')
                    f.write('    \n')
                    f.write(f'    {i} = (tr_sum / {window}) / (rates_array[last_idx].close + epsilon);\n')
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
                    f.write('    double sma_sum = 0.0;\n')
                    f.write('    \n')
                    f.write(f'    for(int i = last_idx - {window} + 1; i <= last_idx; i++)\n')
                    f.write('    {\n')
                    f.write('        sma_sum += rates_array[i].close;\n')
                    f.write('    }\n')
                    f.write(f'    double sma = sma_sum / {window};\n')
                    f.write('    \n')
                    f.write('    double variance = 0.0;\n')
                    f.write(f'    for(int i = last_idx - {window} + 1; i <= last_idx; i++)\n')
                    f.write('    {\n')
                    f.write('        variance += MathPow(rates_array[i].close - sma, 2);\n')
                    f.write('    }\n')
                    f.write(f'    double std = MathSqrt(variance / {window});\n')
                    f.write('    \n')
                    f.write(f'    {i} = (4.0 * std) / (sma + epsilon);\n')

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
                f.write(f'           future_vol[{len(self.models) - 1 - i}] = output[0];\n')
                f.write(f'           past_vol[{len(self.models) - 1 - i}] = 0;\n')
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
        self.X_shape = 0
        self.is_fitted = False
        self.onnx_load = False
        self.early_stopping = early_stopping
        self.V = Visualization('dark')
        if sett == {}:
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

    def save(self, name, type_to_save='default'):
        if type_to_save == 'default':
            os.makedirs(name, exist_ok=True)
            initial_type = [('float_input', FloatTensorType([None, self.X_shape]))]

            file_path = os.path.join(name, f"{name}_features.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.feature_list, f, ensure_ascii=False, indent=2)

            if hasattr(self, 'scaler'):
                scaler_path = os.path.join(name, f"{name}_scaler.pkl")
                joblib.dump(self.scaler, scaler_path)

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

        scaler_files = [f for f in os.listdir(name) if f.endswith('_scaler.pkl')]
        if scaler_files:
            scaler_path = os.path.join(name, scaler_files[0])
            self.scaler = joblib.load(scaler_path)

        model_files = [f for f in os.listdir(name) if f.endswith('.onnx')]

        if not model_files:
            raise FileNotFoundError(f"В папке {name} не найдено .onnx файлов")

        file_path = os.path.join(name, f"{name}_features.json")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.feature_list = json.load(f)

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
        self.X_shape = 0
        self.is_fitted = False
        self.onnx_load = False
        self.early_stopping = early_stopping
        self.V = Visualization('dark')
        if sett == {}:
            self.base_model = xgboost.XGBRegressor(
                objective='reg:squarederror',
                learning_rate=0.1,
                n_estimators=1,
                max_depth=3,
                min_child_weight=5,
                subsample=0.8,
                random_state=42,
            )
        else:
            self.base_model = xgboost.XGBRegressor(**sett)


    def save(self, name, type_to_save='default'):
        if type_to_save == 'default':
            os.makedirs(name, exist_ok=True)
            initial_type = [('float_input', FloatTensorType([None, self.X_shape]))]

            file_path = os.path.join(name, f"{name}_features.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.feature_list, f, ensure_ascii=False, indent=2)

            if hasattr(self, 'scaler'):
                scaler_path = os.path.join(name, f"{name}_scaler.pkl")
                joblib.dump(self.scaler, scaler_path)

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

        file_path = os.path.join(name, f"{name}_features.json")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.feature_list = json.load(f)


        scaler_files = [f for f in os.listdir(name) if f.endswith('_scaler.pkl')]
        if scaler_files:
            scaler_path = os.path.join(name, scaler_files[0])
            self.scaler = joblib.load(scaler_path)

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


class VolClustLightGB(FichEn):
    def __init__(self, sett, early_stopping=True):
        self.models = []
        self.scaler = StandardScaler()
        self.X_shape = 0
        self.is_fitted = False
        self.onnx_load = False
        self.early_stopping = early_stopping
        self.V = Visualization('dark')
        if sett == {}:
            self.base_model = lgb.LGBMRegressor(
                objective='regression',
                learning_rate=0.1,
                n_estimators=1,
                num_leaves= 7,
                min_data_in_leaf=3,
                min_data_in_bin=3,
                seed=42,
                verbosity=-1
            )
        else:
            self.base_model = (lgb.LGBMRegressor(**sett))


    def save(self, name, type_to_save='default'):
        if type_to_save == 'default':
            os.makedirs(name, exist_ok=True)
            initial_type = [('float_input', FloatTensorType([None, self.X_shape]))]

            file_path = os.path.join(name, f"{name}_features.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.feature_list, f, ensure_ascii=False, indent=2)

            if hasattr(self, 'scaler'):
                scaler_path = os.path.join(name, f"{name}_scaler.pkl")
                joblib.dump(self.scaler, scaler_path)

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

        file_path = os.path.join(name, f"{name}_features.json")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.feature_list = json.load(f)

        scaler_files = [f for f in os.listdir(name) if f.endswith('_scaler.pkl')]
        if scaler_files:
            scaler_path = os.path.join(name, scaler_files[0])
            self.scaler = joblib.load(scaler_path)

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
