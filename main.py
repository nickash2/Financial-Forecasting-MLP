from src.preprocess import preprocess, plot_preprocessed

# from src.predict import Predictor
from src.dataset import TimeSeriesDataset
import pandas as pd
from torch.utils.data import DataLoader


def create_dataloader(residuals, window_size, batch_size=64):
    dataset = TimeSeriesDataset(residuals, window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    # Data is preprocessed using the preprocess function
    data = preprocess()
    print(data)
    # Preprocessed img of data is in the plots folder
    # Normalize the data
    plot_preprocessed(data)
    # Take series from the data
    # in a loop, train test split
    # train model & tune
