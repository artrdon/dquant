import numpy as np


def qlike_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    eps = 1e-10
    y_pred = np.clip(y_pred, eps, None)
    loss = np.log(y_pred) + y_true / y_pred
    return np.mean(loss)
