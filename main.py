import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import optuna
import numpy as np
from src.preprocess import preprocess, plot_preprocessed
from src.dataset import TimeSeriesDataset
from src.train import objective, train_final_model
from src.predict import Predictor
from src.mlp import SMAPELoss
from sklearn.metrics import mean_absolute_error
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
        micro_df, test_size=0.2, shuffle=True, random_state=169
    )

    return micro_train_val_df, micro_test_df


def preprocess_data_and_create_dataset(dataset, name, test):
    preprocessed_data = preprocess(dataset, test)
    plot_preprocessed(preprocessed_data, name)
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

    return train_val_data, test_data


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


def calculate_and_print_metrics(
    predictions, test_data, best_params, predictor, train_val_data
):
    predictions = np.array(predictions)
    true_values = torch.tensor([test_data[i][1] for i in range(len(test_data))]).numpy()
    print("Predictions", predictions)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(true_values, predictions)
    print(f"Mean Absolute Error: {mae}")

    # Adjust predictions based on training data trends and average scaler
    scaler = pickle.load(open("data/train_scaler.pkl", "rb"))

    # Print the scaler min_ and scale_ attributes
    print("Scaler min:", scaler.min_)
    print("Scaler scale:", scaler.scale_)

    # Print the shape of predictions and true_values
    print("Predictions shape:", predictions.shape)
    print("True values shape:", true_values.shape)

    # Print the first 10 predictions and true_values before undoing normalization
    print("Raw predictions:", predictions[:400])
    print("Raw true values:", true_values[:400])

    adjusted_predictions = predictor.undo_normalization(predictions, scaler)
    true_values_df = predictor.undo_normalization(true_values, scaler)

    retrended_predictions = predictor.retrend_data(adjusted_predictions)
    retrended_true_values = predictor.retrend_data(true_values_df)

    adjusted_predictions = retrended_predictions
    true_values_df = retrended_true_values

    print("Adjusted Predictions:", adjusted_predictions[:100])
    print("True Values:", true_values_df[:100])

    # Plot adjusted predictions
    time_index = np.arange(700)
    plt.figure(figsize=(10, 5))
    plt.plot(time_index, true_values_df[:700], label="True Values")
    plt.plot(time_index, adjusted_predictions[:700], label="Prediction", linestyle="--")

    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.title("True Values and Adjusted Predictions")
    plt.legend()
    plt.savefig("plots/true_values_and_adjusted_predictions.png")
    # Calculate the smape
    smape_loss = SMAPELoss()
    smape = smape_loss.forward(true_values_df, adjusted_predictions)

    print(f"SMAPE: {smape.item()}%")

    return adjusted_predictions, true_values_df


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_val_data, test_data = load_and_preprocess_data()
    tuning_mode = False  # runs the tuning mode wwith the optuna study
    train_model = True  # trains the final model with hyperparams

    if tuning_mode:
        study = run_tuning_mode(train_val_data, device)
        plot_best_study(study)
    else:
        predictions, best_params, predictor = run_non_tuning_mode(
            train_val_data, test_data, device, train_model
        )
        pred, test = calculate_and_print_metrics(
            predictions, test_data, best_params, predictor, train_val_data
        )
        # plot_raw_data(train_val_data.data, test_data.data)
