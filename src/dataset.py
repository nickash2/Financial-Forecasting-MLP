# This file contains the TimeSeriesDataset class which is used to create a dataset for time series forecasting.
from torch.utils.data import Dataset
import torch


class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size, timeseries_indices):
        self.data = data
        self.window_size = window_size
        self.timeseries_indices = timeseries_indices
        self.current_timeseries = 0

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        if idx + self.window_size >= self.timeseries_indices[self.current_timeseries + 1]:
            self.current_timeseries += 1
            idx = self.timeseries_indices[self.current_timeseries]
        X = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size]
        return X.clone().detach(), y.clone().detach()

# timeseries_indices = list(df.groupby('ID').first().index.values) + [len(df)]
