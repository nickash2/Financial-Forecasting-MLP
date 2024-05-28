from src.preprocess import preprocess, plot_preprocessed

# from src.predict import Predictor
from src.dataset import TimeSeriesDataset
import pandas as pd
from torch.utils.data import DataLoader
import torch

def create_dataloader(residuals, window_size):
    return TimeSeriesDataset(residuals, window_size)


if __name__ == "__main__":
    # Data is preprocessed using the preprocess function
    train, test = preprocess()
    # Preprocessed img of data is in the plots folder
    plot_preprocessed(train)
    loaded_train = create_dataloader(torch.tensor(train['Value'].values), window_size=5)

    print(loaded_train[0])
    # Print the type and shape of the first batch
    # print(first_batch[0].shape, first_batch[1].shape)
    # Take series from the data
    # in a loop, train test split
    # train model & tune
