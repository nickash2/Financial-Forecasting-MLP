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
        micro_df, test_size=0.2, shuffle=False, random_state=169
    )

    return micro_train_val_df, micro_test_df


def preprocess_data_and_create_dataset(dataset, name, test):
    preprocessed_data = preprocess(dataset, test)
    # plot_preprocessed(preprocessed_data, name)
    dataset = TimeSeriesDataset(preprocessed_data, window_size=5)
    return dataset


def create_study_and_pruner():
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource="auto", reduction_factor=3
    )
    study = optuna.create_study(
        study_name="MLP-Tuning-16-06-n",
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
    test_data.window_size = int(best_params["window_size"])
    print(test_data)
    # Prepare test windows for predictions
    test_windows = torch.stack([test_data[i][0] for i in range(len(test_data))])

    # Predict using the loaded model
    predictions = []
    for i in range(len(test_windows)):
        window = test_windows[i].numpy()
        next_prediction = predictor.predict_next(window)
        predictions.append(next_prediction)
        # print(f"{window} => {next_prediction}")

    return predictions, best_params, predictor


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
    initialization_data = test_data.data[:-k]
    true_last_k_points = test_data.data[-k:]

    initial_window = initialization_data[-predictor.model.input_size:]

    predictions = []
    for _ in range(k):
        next_prediction = predictor.predict_next(initial_window)
        predictions.append(next_prediction)
        initial_window = np.append(initial_window[1:], next_prediction)

    predictions = np.array(predictions)

    scaler = pickle.load(open("data/train_scaler.pkl", "rb"))
    last_value = pickle.load(open("data/last_value.pkl", "rb"))

    adjusted_predictions = predictor.undo_normalization(predictions, scaler)
    true_last_k_points = predictor.undo_normalization(true_last_k_points, scaler)
    initialization_data = predictor.undo_normalization(initialization_data, scaler)

    # Use the last value before differencing (for training data) as the starting point for reversal
    adjusted_predictions = predictor.retrend_data(adjusted_predictions, last_value)
    true_last_k_points = predictor.retrend_data(true_last_k_points, last_value)
    initialization_data = predictor.retrend_data(initialization_data, last_value)

    smape_loss = SMAPELoss()
    smape = smape_loss.forward(true_last_k_points, adjusted_predictions)

    print(f"SMAPE for k-step prediction: {smape.item()}%")

    return adjusted_predictions, true_last_k_points, smape.item(), initialization_data




def plot_k_step_predictions(series_N1875_values, true_last_k_points, adjusted_predictions):
    plt.figure(figsize=(10, 5))

    # Plot entire test data without the last 18 points
    plt.plot(series_N1875_values[:-18], color="black", label="Initialization Data")

    # Plot the true last 18 points
    time_index = np.arange(len(series_N1875_values) - 18, len(series_N1875_values))
    plt.plot(time_index, true_last_k_points, color="blue", label="True Values")

    # Plot the predictions
    plt.plot(
        time_index,
        adjusted_predictions,
        color="orange",
        label="Prediction",
        linestyle="--",
    )

    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.title("True Values and k-Step Predictions")
    plt.legend()
    plt.savefig("plots/k_step_predictions.png")



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
        df_test_raw = pd.melt(
            test_data_raw,
            id_vars=[
                "Series",
                "N",
                "NF",
                "Category",
                "Starting Year",
                "Starting Month",
            ],
            var_name="Month",
            value_name="Value",
        )
        # series_N1875_values = df_test_raw[df_test_raw["Series"] == "N1875"][
        #     "Value"
        # ].values
        predictions, best_params, predictor = run_non_tuning_mode(
            train_val_data, test_data, device, train_model
        )

        k = 18  # standard choice in the MP3 competition
        adjusted_predictions, true_last_k_points, smape, initialization_data = (
            k_step_prediction_and_evaluate(
                test_data, k, predictor
            )
        )

        plot_k_step_predictions(initialization_data, true_last_k_points, adjusted_predictions)
