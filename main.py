from src.preprocess import preprocess, plot_preprocessed
from src.dataset import TimeSeriesDataset
import torch
from torch.utils.data import DataLoader
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

    # Create datasets
    train_dataset = TimeSeriesDataset(preprocessed_data, window_size=5, pred_length=1, train=True)
    val_dataset = TimeSeriesDataset(preprocessed_data, window_size=5, pred_length=1, train=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print(f"Length of train_dataset: {len(train_dataset)}")

    # Print the first few items in the dataset
    for i in range(min(5, len(train_dataset))):  # Change 5 to the number of items you want to print
        data = train_dataset[i]
        print(f"Item {i}: {data}")

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, device), n_trials=100)

    # Print the best hyperparameters
    print("Best trial:")
    trial = study.best_trial
    print(" Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


