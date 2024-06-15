import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np
import pickle


def plot_preprocessed(df_detrended, name):
    series_to_plot = df_detrended["Series"].unique()
    # Calculate the number of rows needed for subplots
    n = 9
    ncols = 3
    nrows = n // ncols + n % ncols
    pos = range(1, n + 1)

    fig = plt.figure(figsize=(20, 20))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)

    for k, series in zip(pos, series_to_plot):
        # Filter df_monthly for the current series
        series_data = df_detrended[df_detrended["Series"] == series]

        # Create subplot
        ax = fig.add_subplot(nrows, ncols, k)

        # Plot the series
        ax.plot(series_data["Month"], series_data["Value"])
        ax.set_xlabel("Month")
        ax.set_ylabel("Value")
        ax.set_title(f"Series: {series}")
        ax.grid(True)

    plt.savefig(f"plots/preprocessed_series_{name}.png")



def preprocess(dataset, test=False):
    df_long = pd.melt(
        dataset,
        id_vars=["Series", "N", "NF", "Category", "Starting Year", "Starting Month"],
        var_name="Month",
        value_name="Value",
    )

    series_to_plot = df_long["Series"].unique()
    if not test:
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        with open("data/train_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
    
    df_final = pd.DataFrame()
    if test:
        with open("data/linear_regr.pkl", "rb") as f:
            lasso = pickle.load(f)
    else:
        lasso = Lasso(alpha=2.5, max_iter=6950)

    for series in series_to_plot:
        df_filtered = df_long[df_long["Series"] == series]
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



