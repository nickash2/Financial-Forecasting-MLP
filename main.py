from src.preprocess import preprocess, plot_preprocessed
from src.tune import tune_model
from src.predict import Predictor
from src.dataset import TimeSeriesDataset
import pandas as pd
from torch.utils.data import DataLoader


def create_dataloader(residuals, window_size, batch_size=64):
    dataset = TimeSeriesDataset(residuals, window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    data = preprocess()
    print(data.head())
    plot_preprocessed(data)
