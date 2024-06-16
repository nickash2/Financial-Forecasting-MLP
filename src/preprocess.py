import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np
import pickle


def plot_preprocessed(df_detrended, name):
    plt.figure(figsize=(10, 6))
    time = np.arange(len(df_detrended))
    plt.plot(time, df_detrended["Value"])
    plt.xlabel("Month")
    plt.ylabel("Value")
    plt.title(f"Series: {name}")
    plt.grid(True)
    plt.savefig(f"plots/preprocessed_series_{name}.png")



def preprocess(dataset, test=False):
    df_long = pd.melt(
        dataset,
        id_vars=["Series", "N", "NF", "Category", "Starting Year", "Starting Month"],
        var_name="Month",
        value_name="Value",
    )

    series_to_plot = df_long["Series"].unique()
    
    df_detrended = pd.DataFrame()
    
    if not test:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        lasso = Lasso(alpha=2.5, max_iter=10000)
    else:
        with open("data/train_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        with open("data/linear_regr.pkl", "rb") as f:
            lasso = pickle.load(f)
    
    combined_data = pd.DataFrame()
    for series in series_to_plot:
        df_filtered = df_long[df_long["Series"] == series]
        combined_data = pd.concat([combined_data, df_filtered])

    data = df_long["Value"]
    X = np.arange(len(data)).reshape(-1, 1)
    y = data.values.reshape(-1, 1)

    if not test:
        lasso.fit(X, y)
        with open("data/linear_regr.pkl", "wb") as f:
            pickle.dump(lasso, f)

    fitted_values = lasso.predict(X)
    detrended_data = data - fitted_values.flatten()
    combined_data.loc[:, "Value"] = detrended_data

    df_detrended = combined_data

    combined_values = df_detrended["Value"].values.reshape(-1, 1)
    if not test:
        df_detrended["Value"] = scaler.fit_transform(combined_values)
        with open("data/train_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
    else:
        df_detrended["Value"] = scaler.transform(combined_values)

    return df_detrended




