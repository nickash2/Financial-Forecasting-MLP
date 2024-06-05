from sklearn.model_selection import BaseCrossValidator
import numpy as np

class BlockedTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            if start != 0:  # Skip the first block for training set
                yield np.arange(0, start), np.arange(start, stop)
            current = stop

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
