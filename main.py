from src.preprocess import preprocess, plot_preprocessed
from src.dataset import TimeSeriesDataset
from src.blockedcv import BlockedTimeSeriesSplit
import torch
from torch.utils.data import DataLoader, Subset
from src.train import objective, train_final_model
import optuna
import matplotlib.pyplot as plt
import pickle


def plot_best(trial):
    fig = optuna.visualization.plot_optimization_history(trial)
    fig.write_image("plots/optimization_history.png")


def preprocess_data_and_create_dataset():
    preprocessed_data = preprocess()
    plot_preprocessed(preprocessed_data)
    # print(preprocessed_data)
    dataset = TimeSeriesDataset(preprocessed_data, window_size=5)
    return dataset


def create_study_and_pruner():
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource="auto", reduction_factor=3
    )
    study = optuna.create_study(
        study_name="MLP-Tuning",
        direction="minimize",
        pruner=pruner,
        storage="sqlite:///data/tuning.db",
    )
    return study


def split_data(dataset):
    test_size = int(len(dataset) * 0.2)
    train_val_size = len(dataset) - test_size
    train_val_indices = list(range(train_val_size))
    test_indices = list(range(train_val_size, len(dataset)))
    train_val_data = Subset(dataset, train_val_indices)
    test_data = Subset(dataset, test_indices)
    return train_val_data, test_data


def tuning_mode_operation(dataset, train_val_data, study, device):
    blocked_split = BlockedTimeSeriesSplit(n_splits=5)
    for train_indices, val_indices in blocked_split.split(train_val_data):
        train_subset = Subset(dataset, train_indices.tolist())
        val_subset = Subset(dataset, val_indices.tolist())
        train_loader = DataLoader(
            train_subset, batch_size=32, shuffle=False, drop_last=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=32, shuffle=False, drop_last=True
        )
        study.optimize(
            lambda trial: objective(trial, train_loader, val_loader, device),
            n_trials=50,
        )
    return study


def non_tuning_mode_operation(train_val_data, final_train=False):
    best_params = {}
    with open("habrok_output/06-06/Best_hyperparameters.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if ":" in line:
                key, value = line.strip().split(":")
                best_params[key.strip()] = float(value.strip())
    print(best_params)
    combined_train_val_loader = DataLoader(
        train_val_data, batch_size=32, shuffle=False, drop_last=True
    )
    if final_train:
        print("Training the model with the best hyperparameters...")
        train_final_model(
            train_val_data, combined_train_val_loader, best_params, device
        )
    return best_params, combined_train_val_loader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    dataset = preprocess_data_and_create_dataset()
    train_val_data, test_data = split_data(dataset)
    tuning_mode = False
    if tuning_mode:
        study = create_study_and_pruner()
        study = tuning_mode_operation(dataset, train_val_data, study, device)
        with open("study.pkl", "wb") as f:
            pickle.dump(study, f)
    else:
        best_params, combined_train_val_loader = non_tuning_mode_operation(
            train_val_data, final_train=False
        )
        with open("study.pkl", "rb") as f:
            study = pickle.load(f)
            plot_best(study)
