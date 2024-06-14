import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import optuna

# Define the exponential and polynomial functions
def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

def polynomial_func(x, *coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

# Preprocessing function
def preprocess(dataset, method='difference', degree=2, test=False):
    df_long = pd.melt(
        dataset,
        id_vars=["Series", "N", "NF", "Category", "Starting Year", "Starting Month"],
        var_name="Month",
        value_name="Value",
    )

    df_monthly = df_long[df_long["Category"] == "MICRO       "]
    series_to_plot = df_monthly["Series"].unique()

    if test:
        with open("data/train_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        if method == 'exponential':
            with open("data/exp_params.pkl", "rb") as f:
                trend_params = pickle.load(f)
        elif method == 'polynomial':
            with open("data/poly_params.pkl", "rb") as f:
                trend_params = pickle.load(f)
    else:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        trend_params = {}

    df_final = pd.DataFrame()

    for series in series_to_plot:
        df_filtered = df_monthly[df_monthly["Series"] == series]
        data = df_filtered["Value"].values
        X = np.arange(len(data))

        if method == 'difference':
            detrended_data = np.diff(data, prepend=data[0])
        elif method == 'exponential':
            if not test or series not in trend_params:
                popt, _ = curve_fit(exponential_func, X, data, maxfev=10000)
                trend_params[series] = popt
            else:
                popt = trend_params[series]
            trend_values = exponential_func(X, *popt)
            detrended_data = data / trend_values
        elif method == 'polynomial':
            if not test or series not in trend_params:
                popt, _ = curve_fit(polynomial_func, X, data, p0=[1] * (degree + 1), maxfev=10000)
                trend_params[series] = popt
            else:
                popt = trend_params[series]
            trend_values = polynomial_func(X, *popt)
            detrended_data = data / trend_values

        df_filtered.loc[:, "Value"] = detrended_data

        if not test:
            df_filtered.loc[:, "Value"] = scaler.fit_transform(
                df_filtered["Value"].values.reshape(-1, 1)
            )
        else:
            df_filtered.loc[:, "Value"] = scaler.transform(
                df_filtered["Value"].values.reshape(-1, 1)
            )
        df_final = pd.concat([df_final, df_filtered])
    
    if not test:
        with open("data/train_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        if method == 'exponential':
            with open("data/exp_params.pkl", "wb") as f:
                pickle.dump(trend_params, f)
        elif method == 'polynomial':
            with open("data/poly_params.pkl", "wb") as f:
                pickle.dump(trend_params, f)
    
    return df_final

# Objective function for Optuna study
def objective(trial, train_data_raw, val_data_raw):
    method = trial.suggest_categorical('method', ['difference', 'exponential', 'polynomial'])
    degree = trial.suggest_int('degree', 1, 5) if method == 'polynomial' else 2
    
    # Preprocess training and validation data
    train_data = preprocess(train_data_raw, method=method, degree=degree, test=False)
    val_data = preprocess(val_data_raw, method=method, degree=degree, test=True)

    # Evaluate the method based on Mean Absolute Error
    y_val_true = val_data["Value"].values.reshape(-1, 1)
    mae = mean_absolute_error(y_val_true, val_data["Value"].values.reshape(-1, 1))
    
    return mae

if __name__ == "__main__":
    # Load the raw data
    raw_data = pd.read_excel("data/M3C.xls", sheet_name="M3Month")
    raw_data.dropna(axis=1, inplace=True)
    
    # Split the data into train, validation, and test sets
    train_data_raw, test_data_raw = train_test_split(raw_data, test_size=0.2, random_state=42)
    train_data_raw, val_data_raw = train_test_split(train_data_raw, test_size=0.2, random_state=42)

    # Create an Optuna study and optimize it to find the best method and degree
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_data_raw, val_data_raw), n_trials=100)

    # Get the best hyperparameters and MAE found by Optuna
    best_method = study.best_params['method']
    best_degree = study.best_params['degree']
    best_mae = study.best_value

    print("Best method:", best_method)
    print("Best degree:", best_degree)
    print("Best MAE:", best_mae)

    # Optionally, preprocess test data using the best method and degree
    test_data = preprocess(test_data_raw, method=best_method, degree=best_degree, test=True)

    # Optionally, plot preprocessed data for the best method on test set
    plot_preprocessed(test_data, best_method + "_test")
