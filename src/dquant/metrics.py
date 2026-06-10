import numpy as np


def qlike_score(y_true, y_pred):
    sigma2_true = y_true
    sigma2_pred = np.maximum(y_pred, 1e-10)
    return np.mean(np.log(sigma2_pred) + sigma2_true / sigma2_pred)

