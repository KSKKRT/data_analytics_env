import numpy as np

from .setting import seed_everything

seed_everything(seed=42)


class MapK:
    def __init__(self, y_true, y_pred, K):
        self.true = y_true
        self.pred = y_pred
        self.K = K

    def _apk(self, y_i_true, y_i_pred):
        assert len(y_i_pred) <= self.K
        assert len(np.unique(y_i_pred)) == len(y_i_pred)

        sum_precision = 0.0
        num_hits = 0.0

        for i, p in enumerate(y_i_pred):
            if p in y_i_true:
                num_hits += 1
                precision = num_hits / (i + 1)
                sum_precision += precision

        return sum_precision / min(len(y_i_true), self.K)

    def mapk(self):
        return np.mean(
            [
                self._apk(y_i_true, y_i_pred)
                for y_i_true, y_i_pred in zip(self.true, self.pred)
            ]
        )
