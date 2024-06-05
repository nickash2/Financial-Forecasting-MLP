# This file contains the TimeSeriesDataset class which is used to create a dataset for time series forecasting.
from torch.utils.data import Dataset
import torch

# Can mention in the report that after detrending, 
# the data is randomised and dont need to keep track of the timeseries 
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size, timeseries_indices):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        X = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size]
        return X.clone().detach(), y.clone().detach()
