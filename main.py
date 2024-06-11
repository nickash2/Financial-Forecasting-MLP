from src.preprocess import preprocess, plot_preprocessed
from src.dataset import TimeSeriesDataset
from src.blockedcv import BlockedTimeSeriesSplit
import torch
from torch.utils.data import DataLoader, Subset
from src.train import objective, train_final_model
import optuna
import matplotlib.pyplot as plt
import pickle
from src.predict import Predictor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



def plot_best(trial):
    fig = optuna.visualization.plot_optimization_history(trial)
    fig.write_image("plots/optimization_history.png")



def add_trend_back(series_detrended, series_name, series_info):
    # Get the trend, mean, and sd for this series
    info = series_info[series_name]
    trend = info['trend']
    mean = info['mean']
    sd = info['sd']

    # Add the trend back to the series
    series = (series_detrended * sd) + mean + trend

    return series

def preprocess_data_and_create_dataset(dataset, name):
    preprocessed_data, series_info = preprocess(dataset)
    # print(preprocessed_data)
    plot_preprocessed(preprocessed_data, name)
    dataset = TimeSeriesDataset(preprocessed_data, window_size=5)
    return dataset, series_info


def create_study_and_pruner():
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource="auto", reduction_factor=3
    )
    study = optuna.create_study(
        study_name="MLP-Tuning-12-06L",
        direction="minimize",
        pruner=pruner,
        storage="sqlite:///data/tuning.db",
        load_if_exists=True,
    )
    return study


def split_data(df):
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_val_df, test_df


def tuning_mode_operation(dataset, study, device):
    study.optimize(
            lambda trial: objective(trial, dataset, device),
            n_trials=1000,
    )
    return study

def load_data():
    df = pd.read_excel("data/M3C.xls", sheet_name="M3Month")
    df.dropna(axis=1, inplace=True)
    return df


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

    raw_data = load_data()

    train_val_data_raw, test_data_raw = split_data(raw_data)

    train_val_data, train_series_info = preprocess_data_and_create_dataset(train_val_data_raw, "train")
    test_data, test_series_info = preprocess_data_and_create_dataset(test_data_raw, "test")
    # print(test_series_info)
    tuning_mode = True
    if tuning_mode:
        study = create_study_and_pruner()
        study = tuning_mode_operation(train_val_data, study, device)

    else:
        best_params, combined_train_val_loader = non_tuning_mode_operation(
            train_val_data, final_train=False
        )

        # Do predictions
        predictor = Predictor(model_path="models/final_model.pth", device=device)

        test_windows = torch.stack([test_data[i][0] for i in range(len(test_data))])

        predictions = []
        for i in range(len(test_windows)):
            window = test_windows[i].numpy()
            next_prediction = predictor.predict_next(window)
            predictions.append(next_prediction)
            print(f"Window {i}: {window} -> Next predicted point: {next_prediction}")

        predictions = np.array(predictions)
        true_values = torch.tensor([test_data[i][1] for i in range(len(test_data))]).numpy()

        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(true_values, predictions)
        print(f"Mean Absolute Error: {mae}")


        # Create a plot
        plt.figure(figsize=(10,6))
        plt.plot(true_values, label='True Values')
        plt.plot(predictions, label='Predictions')
        plt.title('Predictions vs True Values')
        plt.xlabel('Observation')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

        # # Make empty arrays to store the predictions and true values with the trend added back
        # predictions_with_trend = np.empty_like(predictions)
        # true_values_with_trend = np.empty_like(true_values)
        
        # series_names = list(test_series_info.keys())

        # for i, series_name in enumerate(series_names):
        #     # Add the trend back to the predictions and true values for this series
        #     predictions_with_trend[i] = add_trend_back(predictions[i], series_name, test_series_info)
        #     true_values_with_trend[i] = add_trend_back(true_values[i], series_name, test_series_info)

        
        # # Plot predictions vs actual values
        # plt.figure(figsize=(10, 6))
        # plt.plot(true_values_with_trend, label='Actual', color='blue')
        # plt.plot(predictions_with_trend, label='Predicted', color='red')
        # plt.title('Test Data: Actual vs Predicted')
        # plt.xlabel('Time')
        # plt.ylabel('Value')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # Calculate the SMAPE
        # smape_value = predictor.accuracy(true_values_with_trend, predictions_with_trend)
        # print(f"SMAPE: {smape_value:.2f}%")
