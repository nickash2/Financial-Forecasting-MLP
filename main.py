import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import optuna
import numpy as np
from src.preprocess import preprocess
from src.dataset import TimeSeriesDataset
from src.train import objective, train_final_model
from src.predict import Predictor
from src.mlp import SMAPELoss
import pickle


def plot_best_study(trial):
    fig = optuna.visualization.plot_optimization_history(trial)
    fig.write_image("plots/optimization_history.png")


def load_data():
    df = pd.read_excel("data/M3C.xls", sheet_name="M3Month")
    df.dropna(axis=1, inplace=True)
    return df


def split_data(df):
    # Filter for the "MICRO" category
    micro_df = df[df["Category"].str.strip() == "MICRO"]

    # Split the micro category data into train and test sets
    micro_train_val_df, micro_test_df = train_test_split(
        micro_df, test_size=0.2, shuffle=True, random_state=222
    )

    return micro_train_val_df, micro_test_df


def preprocess_data_and_create_dataset(dataset, name, test):
    preprocessed_data = preprocess(dataset, test)
    if not test:
        dataset = TimeSeriesDataset(preprocessed_data, window_size=5)
        return dataset
    else:
        return preprocessed_data


def create_study_and_pruner():
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource="auto", reduction_factor=3
    )
    study = optuna.create_study(
        study_name="MLP-Tuning-17-06",
        direction="minimize",
        pruner=pruner,
        storage="sqlite:///data/tuning.db",
        load_if_exists=True,
    )
    return study


def tuning_mode_operation(dataset, study, device, n_trials=200):
    study.optimize(
        lambda trial: objective(trial, dataset, device),
        n_trials=n_trials,
    )
    return study


def non_tuning_mode_operation(train_val_data, final_train=False):
    best_params = {}
    with open("habrok_output/12-06/Best_hyperparameters.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if ":" in line:
                key, value = line.strip().split(":")
                best_params[key.strip()] = float(value.strip())

    train_val_data.window_size = int(best_params["window_size"])
    combined_train_val_loader = DataLoader(
        train_val_data, batch_size=32, shuffle=False, drop_last=True
    )
    if final_train:
        print("Training the model with the best hyperparameters...")
        train_final_model(combined_train_val_loader, best_params, device)
    return best_params


def load_and_preprocess_data():
    raw_data = load_data()
    train_val_data_raw, test_data_raw = split_data(raw_data)

    train_val_data = preprocess_data_and_create_dataset(
        train_val_data_raw, "train", test=False
    )

    test_data = preprocess_data_and_create_dataset(test_data_raw, "test", test=True)

    return train_val_data, test_data, test_data_raw


def run_tuning_mode(train_val_data, device):
    study = create_study_and_pruner()
    study = tuning_mode_operation(train_val_data, study, device)
    with open("habrok_output/12-06/Best_hyperparameters.txt", "w") as f:
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")
    return study


def run_non_tuning_mode(train_val_data, test_data, device, train_model):
    best_params = non_tuning_mode_operation(train_val_data, final_train=train_model)
    print(best_params)

    # Load the predictor model for making predictions
    predictor = Predictor(best_params=best_params, device=device)

    return best_params, predictor


def plot_raw_data(train_data, test_data):
    plt.figure(figsize=(12, 6))
    plt.plot(train_data, label="Raw Training Data", color="blue")
    plt.plot(test_data, label="Raw Test Data", color="orange")
    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.title("Raw Training and Test Data Comparison")
    plt.legend()
    plt.show()


def k_step_prediction_and_evaluate(test_data, k, predictor):
    initialization_data = test_data.dropna().values[:-k]
    true_last_k_points = test_data.dropna().values[-k:]

    initial_window = initialization_data[-predictor.model.input_size:]

    predictions = []
    for _ in range(k):
        next_prediction = predictor.predict_next(initial_window)
        predictions.append(next_prediction)
        initial_window = np.append(initial_window[1:], next_prediction)

    predictions = np.array(predictions)

    scaler = pickle.load(open("data/train_scaler.pkl", "rb"))
    last_value = pickle.load(open("data/last_value.pkl", "rb"))


    # Undo normalization
    initialization_data_unnorm = predictor.undo_normalization(initialization_data, scaler)
    predictions_unnorm = predictor.undo_normalization(predictions, scaler)
    true_last_k_points_unnorm = predictor.undo_normalization(true_last_k_points, scaler)


    # Retrend the initialization data
    initialization_data_retrended = np.cumsum(initialization_data_unnorm) + last_value

    # Use the last value from the initialization data retrended
    last_known_value = initialization_data_retrended[-1]

    # Retrend the predictions
    adjusted_predictions = np.cumsum(predictions_unnorm) + last_known_value

    # Retrend the true last k points
    last_value_for_true_points = initialization_data_retrended[-1]
    true_last_k_points_retrended = np.cumsum(true_last_k_points_unnorm) + last_value_for_true_points

    # Calculate SMAPE
    smape_loss = SMAPELoss()
    smape = smape_loss.forward(true_last_k_points_retrended, adjusted_predictions)

    print(f"SMAPE for k-step prediction: {smape.item()}%")

    return adjusted_predictions, true_last_k_points_retrended, smape.item(), initialization_data_retrended


def plot_k_step_predictions(init_data_retrended, true_last_k_points_unnorm, adjusted_predictions, series, smape):
    plt.figure(figsize=(10, 5))

    # Plot entire test data without the last 18 points
    plt.plot(np.arange(len(init_data_retrended)), init_data_retrended, color="black", label="Original Test Data")

    # Plot the true last 18 points
    time_index_true = np.arange(len(init_data_retrended), len(init_data_retrended) + len(true_last_k_points_unnorm))
    plt.plot(time_index_true, true_last_k_points_unnorm, color="blue", label="True Values")

    time_index_pred = np.arange(len(init_data_retrended), len(init_data_retrended) + len(adjusted_predictions))
    plt.plot(time_index_pred, adjusted_predictions, color="orange", label="Prediction", linestyle="--")

    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.title(f"True Values and k-Step Predictions for {series}, with SMAPE: {smape:.2f}%")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/predictions/k_step_predictions_{series}.png")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_val_data, test_data, test_data_raw = load_and_preprocess_data()
    tuning_mode = False  # runs the tuning mode with the optuna study
    train_model = False  # trains the final model with hyperparams

    if tuning_mode:
        study = run_tuning_mode(train_val_data, device)
        plot_best_study(study)
    else:
        
        best_params, predictor = run_non_tuning_mode(
            train_val_data, test_data, device, train_model
        )

        k = 18  # standard choice in the competition
        # Get all unique series
        print(test_data)
        all_series = test_data["Series"].unique() # type: ignore

        # Initialize a list to store all SMAPE values
        all_smape = []

        # Loop through all series
        for series in all_series:
            # Filter the data for the current series
            test_data_filtered = test_data[test_data["Series"] == series]

            # Run the k_step_prediction_and_evaluate function
            adjusted_predictions, true_last_k_points, smape, initialization_data = (
                k_step_prediction_and_evaluate(
                    test_data_filtered["Value"], k, predictor # type: ignore
                )
            )
            plot_k_step_predictions(initialization_data, true_last_k_points, adjusted_predictions, series=series, smape=smape)

            # Append the SMAPE value to the list
            all_smape.append(smape)

        # Calculate the average SMAPE
        average_smape = sum(all_smape) / len(all_smape)
        print("Avg SMAPE:", average_smape)
