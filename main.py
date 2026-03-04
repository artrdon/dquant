from time import sleep
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd




class VolClust:

    def __init__(self, model_type):
        self.__X = []
        self.__Y = []
        if model_type == 'GB':
            self.model = GradientBoostingRegressor(
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
            raise 'Init Error: Undefined model name. It must be only: GB, XGB, LGB, CB.'

    def __FichEngine(self, data, input_bars, output_bars):
        x_data = []
        y_data = []

        for i in range(len(data)):
            if i < input_bars:
                continue
            elif i >= (len(data) - output_bars):
                break
            else:
                x_data.append(data[i-input_bars:i])
                y_data.append(data[i])

        return x_data, y_data

    def fit(self, data, input_bars, output_bars, trees_count, show_results=False):

        self.__X, self.__Y = self.__FichEngine(data, input_bars, output_bars)
        X_train, X_test, y_train, y_test = train_test_split(self.__X, self.__Y, test_size=0.2)
        train_errors = []
        val_errors = []

        for i in range(1, trees_count+1):
            self.model.n_estimators = i
            self.model.fit(X_train, y_train)
            train_errors.append(mean_squared_error(y_train, self.model.predict(X_train)))
            val_errors.append(mean_squared_error(y_test, self.model.predict(X_test)))

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

    def forecast(self, latest_data):
        return self.model.predict(latest_data)




