from src.preprocess import preprocess, plot_preprocessed
from src.dataset import TimeSeriesDataset
from src.blockedcv import BlockedTimeSeriesSplit
import torch
from torch.utils.data import DataLoader, Subset
from src.train import objective, train_final_model
import optuna
import matplotlib.pyplot as plt
import pickle


INPUT_SIZE = 5  # window size, can be adjusted to any value if needed
OUTPUT_SIZE = 1  # next point

def plot_best(trial):
    fig = optuna.visualization.plot_optimization_history(trial)
    fig.write_image("plots/optimization_history.png")


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
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource="auto", reduction_factor=3
    )

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

    tuning_mode = True

    if tuning_mode:
        blocked_split = BlockedTimeSeriesSplit(n_splits=5)

        for train_indices, val_indices in blocked_split.split(train_val_data):
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)

            train_loader = DataLoader(
                train_subset, batch_size=32, shuffle=False, drop_last=True
            )
            val_loader = DataLoader(
                val_subset, batch_size=32, shuffle=False, drop_last=True
            )

            print(f"Train indices: {train_indices}")
            print(f"Val indices: {val_indices}")

            study.optimize(
                lambda trial: objective(trial, train_loader, val_loader, device),
                n_trials=50,
            )

        # Print the best hyperparameters
        print("Best hyperparameters:")
        print(study.best_params)
        print("Best value:")
        print(study.best_value)
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")

        with open("Best_hyperparameters.txt", "w") as f:
            f.write("Best trial\n")
            trial = study.best_trial
            f.write(f" Value {trial.value}\n")
            f.write(" Params \n")
            for key, value in trial.params.items():
                f.write(f"    {key}: {value}\n")

        with open("study.pkl", "wb") as f:
            pickle.dump(study, f)
    else:
        best_params = {}
        with open("habrok_output/06-06/Best_hyperparameters.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if ":" in line:
                    key, value = line.strip().split(":")
                    best_params[key.strip()] = float(
                        value.strip()
                    )  # assuming all parameters are float
        print(best_params)
        # Train the model with the best hyperparameters
        print("Training the model with the best hyperparameters...")
        # Combine training and validation sets for final training
        combined_train_val_loader = DataLoader(train_val_data, batch_size=32, shuffle=False, drop_last=True)
        # train_final_model(train_val_data, combined_train_val_loader, best_params, device)
        with open("study.pkl", "rb") as f:
            study = pickle.load(f)
            plot_best(study)
