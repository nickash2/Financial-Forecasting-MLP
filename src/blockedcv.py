from sklearn.model_selection import BaseCrossValidator
import numpy as np

class BlockedTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits=5):
        super().__init__()
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        if self.n_splits > n_samples:
            raise ValueError("Cannot have number of splits greater than the number of samples.")

        fold_sizes = np.array_split(np.arange(n_samples), self.n_splits)
        for i in range(1, self.n_splits):  # Skip the first block for training set
            train_idx = np.concatenate(fold_sizes[:i])
            val_idx = fold_sizes[i]
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
