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
    

    # Create dataset
    dataset = TimeSeriesDataset(preprocessed_data, window_size=5)
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource='auto', reduction_factor=3)

    study = optuna.create_study(direction="minimize", pruner=pruner)
    # split first to make a test set, maybe use timeseriessplit
    
    # Define the size of your test set
    test_size = int(len(dataset) * 0.2)  # 20% of data for testing
    train_val_size = len(dataset) - test_size

    # Split the data into training+validation set and test set
    
    # Create indices for the splits
    train_val_indices = list(range(train_val_size))
    test_indices = list(range(train_val_size, len(dataset)))

    train_val_data = Subset(dataset, train_val_indices)
    test_data = Subset(dataset, test_indices)

    blocked_split = BlockedTimeSeriesSplit(n_splits=5)

    for train_indices, val_indices in blocked_split.split(train_val_data):
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=False, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, drop_last=True)
        
        print(f"Train indices: {train_indices}")
        print(f"Val indices: {val_indices}")

        study.optimize(lambda trial: objective(trial, train_loader, val_loader, device), n_trials=50)

    # Print the best hyperparameters

    print("Best hyperparameters:")
    print(study.best_params)
    print("Best value:")
    print(study.best_value)
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    with open("Best_hyperparameters.txt", "w") as f:
        f.write("Best trial:\n")
        trial = study.best_trial
        f.write(f" Value: {trial.value}\n")
        f.write(" Params: \n")
        for key, value in trial.params.items():
            f.write(f"    {key}: {value}\n")


# use test set with smape measure for accuracy
# retrend the data for this