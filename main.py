from src.preprocess import preprocess, plot_preprocessed
from src.dataset import TimeSeriesDataset
from src.blockedcv import BlockedTimeSeriesSplit
import torch
from torch.utils.data import DataLoader, Subset
from src.train import objective
import optuna

if __name__ == "__main__":

    # Data is preprocessed using the preprocess function
    preprocessed_data = preprocess()
    # Preprocessed img of data is in the plots folder
    plot_preprocessed(preprocessed_data)
    print(preprocessed_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    blocked_split = BlockedTimeSeriesSplit(n_splits=5)

    # Create dataset
    dataset = TimeSeriesDataset(preprocessed_data, window_size=5)
    
    study = optuna.create_study(direction="minimize")
    
    for train_indices, val_indices in blocked_split.split(dataset):
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=False, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, drop_last=True)
        
        print(f"Train indices: {train_indices}")
        print(f"Val indices: {val_indices}")

        study.optimize(lambda trial: objective(trial, train_loader, val_loader, device), n_trials=50)

    # Print the best hyperparameters
    print("Best trial:")
    trial = study.best_trial
    print(" Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
