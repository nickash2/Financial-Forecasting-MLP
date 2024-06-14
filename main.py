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
from sklearn.metrics import mean_absolute_error
import pickle

def plot_best(trial):
    fig = optuna.visualization.plot_optimization_history(trial)
    fig.write_image("plots/optimization_history.png")

def load_data():
    df = pd.read_excel("data/M3C.xls", sheet_name="M3Month")
    df.dropna(axis=1, inplace=True)
    return df

def split_data(df):
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_val_df, test_df

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
        study_name="MLP-Tuning-14-06-N6",
        direction="minimize",
        pruner=pruner,
        storage="sqlite:///data/tuning.db",
        load_if_exists=True,
    )
    return study

def tuning_mode_operation(dataset, study, device):
    study.optimize(
        lambda trial: objective(trial, dataset, device),
        n_trials=500,
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
    print(best_params)
    
    # Debug statement to check train_val_data type
    print(f"train_val_data type: {type(train_val_data)}")
    
    train_val_data.window_size = int(best_params["window_size"])
    combined_train_val_loader = DataLoader(
        train_val_data, batch_size=32, shuffle=False, drop_last=True
    )
    if final_train:
        print("Training the model with the best hyperparameters...")
        train_final_model(
            combined_train_val_loader, best_params, device
        )
    return best_params, combined_train_val_loader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    raw_data = load_data()

    train_val_data_raw, test_data_raw = split_data(raw_data)

    train_val_data = preprocess_data_and_create_dataset(
        train_val_data_raw, "train", test=False
    )

    test_data = preprocess_data_and_create_dataset(
        test_data_raw, "test", test=True
    )

    tuning_mode = True
    if tuning_mode:
        study = create_study_and_pruner()
        study = tuning_mode_operation(train_val_data, study, device)
        with open("habrok_output/12-06/Best_hyperparameters.txt", "w") as f:
            for key, value in study.best_params.items():
                f.write(f"{key}: {value}\n")

    else:
        best_params, combined_train_val_loader = non_tuning_mode_operation(
            train_val_data, final_train=False
        )
        print(best_params)

        # Load the predictor model for making predictions
        predictor = Predictor(best_params=best_params, device=device)
        test_data.window_size = int(best_params["window_size"])

        # Prepare test windows for predictions
        test_windows = torch.stack([test_data[i][0] for i in range(len(test_data))])

        # Predict using the loaded model
        predictions = []
        for i in range(len(test_windows)):
            window = test_windows[i].numpy()
            next_prediction = predictor.predict_next(window)
            predictions.append(next_prediction)
            # print(f"Window {i}: {window} -> Next predicted point: {next_prediction}")

        predictions = np.array(predictions)
        true_values = torch.tensor([test_data[i][1] for i in range(len(test_data))]).numpy()
        print(predictions)

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
        print("Raw predictions:", predictions[:10])
        print("Raw true values:", true_values[:10])

        adjusted_predictions = predictor.undo_normalization(predictions, scaler)
        true_values_df = predictor.undo_normalization(true_values, scaler)

        retrended_prediction = predictor.retrend_data(adjusted_predictions)
        retrend_test = predictor.retrend_data(true_values_df)


        print("Adjusted Predictions:", adjusted_predictions[:10])
        print("True Values:", true_values_df[:10])

        # Plot adjusted predictions
        time_index = np.arange(10)
        plt.figure(figsize=(10, 5))
        plt.plot(time_index, retrend_test[:10], label="True Values")
        plt.plot(time_index, retrended_prediction[:10], label="Adjusted Predictions", linestyle="--")
        plt.xlabel("Time Index")
        plt.ylabel("Value")
        plt.title("True Values and Adjusted Predictions")
        plt.legend()
        plt.savefig("plots/true_values_vs_adjusted_predictions.png")
        