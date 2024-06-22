import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle

def plot_preprocessed(df_detrended, name, train=False):
    if train:
        plt.figure(figsize=(10, 6))
        time = np.arange(len(df_detrended))
        plt.plot(time, df_detrended)
        plt.xlabel("Month")
        plt.ylabel("Value")
        plt.title(f"Series: {name}")
        plt.grid(True)
        plt.savefig(f"plots/preprocessed_series_{name}.png")
    else:
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

    combined_data = pd.DataFrame()
    for series in df_long["Series"].unique():
        df_filtered = df_long[df_long["Series"] == series]
        combined_data = pd.concat([combined_data, df_filtered])

    if not test:
        # Training phase
        data = combined_data["Value"]
        plot_preprocessed(data, "orig_train", train=True)

        # Perform differencing
        differenced_data = data.diff().dropna()
        
        # Store the last value of the original data before differencing
        last_value = data.iloc[-1]

        combined_data = combined_data.iloc[1:]  # Adjust the index after diff()
        combined_data.loc[:, "Value"] = differenced_data

        # Fit scaler and transform the data
        scaler = MinMaxScaler(feature_range=(-1, 1))
        combined_values = combined_data["Value"].values.reshape(-1, 1) # type: ignore
        combined_data["Value"] = scaler.fit_transform(combined_values)

        # Save the fitted scaler and the last value
        with open("data/train_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open("data/last_value.pkl", "wb") as f:
            pickle.dump(last_value, f)

        plot_preprocessed(combined_data["Value"], "train", train=True)

    else:
        # Testing phase
        try:
            with open("data/train_scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            with open("data/last_value.pkl", "rb") as f:
                last_value = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Scaler or last_value file not found. Ensure the model is trained before testing."
            )

        combined_data = pd.DataFrame()
        series_to_plot = df_long["Series"].unique()
        for series in series_to_plot:
            # Filter the DataFrame for the current series
            df_filtered = df_long[df_long["Series"] == series]

            # Get the data
            data = df_filtered["Value"]

            # Perform differencing
            differenced_data = data.diff().dropna()
            detrended_data = differenced_data

            # Replace the 'Value' column with the detrended data
            df_filtered.loc[:, "Value"] = detrended_data

            # Scale the 'Value' column of the data
            df_filtered.loc[:, "Value"] = scaler.transform(
                df_filtered["Value"].values.reshape(-1, 1)
            )
            combined_data = pd.concat([combined_data, df_filtered])
        # Reset the index of the combined_data DataFrame
        combined_data.reset_index(drop=True, inplace=True)  
        plot_preprocessed(combined_data, "test",train=False)

    return combined_data
