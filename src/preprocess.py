import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
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

    # *As* the data is in wide format, we convert it to a long format.
    df_long = pd.melt(
        dataset,
        id_vars=["Series", "N", "NF", "Category", "Starting Year", "Starting Month"],
        var_name="Month",
        value_name="Value",
    )

    # To pick our group of interest, we decided to pick the "Micro", as it has a lot more data.

    df_monthly = df_long[df_long["Category"] == "MICRO       "]

    series_to_plot = df_monthly["Series"].unique()
    if not test:  # if we are training
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else: # if we are testing
        with open("data/train_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
    # Create an empty DataFrame to store the detrended and de-seasonalized data
    df_final = pd.DataFrame()

    # Create a LinearRegression object
    if test:
        with open("data/linear_regr.pkl", "rb") as f:
            ridge = pickle.load(f)
    else:
        ridge = Ridge(alpha=1.0)

    for series in series_to_plot:
        # Filter the DataFrame for the current series
        df_filtered = df_monthly[df_monthly["Series"] == series]

        # Get the data
        data = df_filtered["Value"]

        # Fit the linear model to the data
        X = np.arange(len(data)).reshape(-1, 1)
        y = data.values.reshape(-1, 1)

        if not test:
            ridge.fit(X, y)

        # Calculate the fitted values
        fitted_values = ridge.predict(X)

        # Subtract the fitted values from the original data to get the detrended data
        detrended_data = data - fitted_values.flatten()

        # Replace the 'Value' column with the detrended data
        df_filtered.loc[:, "Value"] = detrended_data

        # Scale the 'Value' column of the data
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
            pickle.dump(ridge, f)
        with open("data/train_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
    
    
    return df_final



