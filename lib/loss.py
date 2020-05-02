import numpy as np


def cross_entropy_loss(y_true, y_pred, reduction='mean'):
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss) if reduction == 'mean' else loss


def mse_loss(y_true, y_pred, reduction='mean'):
    loss = (y_true - y_pred) ** 2
    return np.mean(loss) if reduction == 'mean' else loss
