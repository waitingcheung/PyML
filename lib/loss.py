import numpy as np


def mse_loss(y_true, y_pred, reduction='mean'):
    loss = (y_true - y_pred) ** 2
    return np.mean(loss) if reduction == 'mean' else loss
