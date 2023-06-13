import numpy as np


class Loss:
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()


class LogLoss(Loss):

    def loss(self, y_true, y_pred):
        raise NotImplementedError()

    def gradient(self, y_true, y_pred):
        raise NotImplementedError()


class SquaredError(Loss):

    def loss(self, y_true, y_pred):
        return 0.5 * np.power((y_true - y_pred), 2)

    def gradient(self, y_true, y_pred):
        return y_true - y_pred
