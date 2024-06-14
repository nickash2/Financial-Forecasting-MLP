import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
import numpy as np
import pickle
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def plot_preprocessed(df_detrended, name):
    series_to_plot = df_detrended["Series"].unique()
    n = min(9, len(series_to_plot))  # Adjusted to plot up to 9 series
    ncols = 3
    nrows = n // ncols + n % ncols
    pos = range(1, n + 1)

    fig = plt.figure(figsize=(20, 20))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)

    for k, series in zip(pos, series_to_plot):
        series_data = df_detrended[df_detrended["Series"] == series]
        ax = fig.add_subplot(nrows, ncols, k)
        ax.plot(series_data["Month"], series_data["Value"])
        ax.set_xlabel("Month")
        ax.set_ylabel("Value")
        ax.set_title(f"Series: {series}")
        ax.grid(True)

    plt.savefig(f"plots/preprocessed_series_{name}.png")

def preprocess(dataset,max_iter, alpha=1.0, test=False):
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
        with open("data/linear_regr.pkl", "rb") as f:
            lasso = pickle.load(f)
    else:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        lasso = Lasso(alpha=alpha, max_iter=max_iter)

    df_final = pd.DataFrame()

    for series in series_to_plot:
        df_filtered = df_monthly[df_monthly["Series"] == series]
        data = df_filtered["Value"]
        X = np.arange(len(data)).reshape(-1, 1)
        y = data.values.reshape(-1, 1)

        if not test:
            lasso.fit(X, y)

        fitted_values = lasso.predict(X)
        detrended_data = data - fitted_values.flatten()
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
        with open("data/linear_regr.pkl", "wb") as f:
            pickle.dump(lasso, f)
        with open("data/train_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
    
    return df_final

def objective(trial, train_data_raw, val_data_raw):
    alpha = trial.suggest_float('alpha', 1e-4, 1e2, log=True)
    max_iter = trial.suggest_int('max_iter', 1000, 10000, log=False,step=10)
    
    df_train = preprocess(train_data_raw, alpha=alpha, test=False, max_iter=max_iter)
    X_train = np.arange(len(df_train)).reshape(-1, 1)
    y_train = df_train["Value"].values.reshape(-1, 1)
    
    df_valid = preprocess(val_data_raw, alpha=alpha, test=True, max_iter=max_iter)
    X_valid = np.arange(len(df_valid)).reshape(-1, 1)
    y_valid = df_valid["Value"].values.reshape(-1, 1)
    
    lasso = Lasso(alpha=alpha, max_iter=max_iter)
    lasso.fit(X_train, y_train)
    preds = lasso.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds)
    
    return mae

if __name__ == "__main__":
    raw_data = pd.read_excel("data/M3C.xls", sheet_name="M3Month")
    raw_data.dropna(axis=1, inplace=True)
    train_data_raw, test_data_raw = train_test_split(raw_data, test_size=0.2, random_state=42)
    train_data_raw, val_data_raw = train_test_split(train_data_raw, test_size=0.2, random_state=42)

    study = optuna.create_study(direction="minimize", study_name="lasso_tuning", storage="sqlite:///data/lasso_tuning.db", load_if_exists=True)
    study.optimize(lambda trial: objective(trial, train_data_raw, val_data_raw), n_trials=100)
    
    print("Best hyperparameters: ", study.best_params)
    print("Best MAE: ", study.best_value)
    
    best_alpha = study.best_params['alpha']
    train_data = preprocess(train_data_raw, alpha=best_alpha, test=False)
    best_lasso = Lasso(alpha=best_alpha)
    best_lasso.fit(np.arange(len(train_data)).reshape(-1, 1), train_data["Value"].values.reshape(-1, 1))
    
    with open("data/best_lasso_model.pkl", "wb") as f:
        pickle.dump(best_lasso, f)
    
    df_test = preprocess(test_data_raw, alpha=best_alpha, test=True)
    X_test = np.arange(len(df_test)).reshape(-1, 1)
    y_test = df_test["Value"].values.reshape(-1, 1)
    
    preds_test = best_lasso.predict(X_test)
    test_mae = mean_absolute_error(y_test, preds_test)
    print(f"Test MAE: {test_mae}")
