import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle

def plot_preprocessed(df_detrended, name):
    plt.figure(figsize=(10, 6))
    time = np.arange(len(df_detrended))
    plt.plot(time, df_detrended)
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

    combined_data = pd.DataFrame()
    for series in df_long["Series"].unique():
        df_filtered = df_long[df_long["Series"] == series]
        combined_data = pd.concat([combined_data, df_filtered])

    if not test:
        # Training phase
        data = combined_data["Value"]
        plot_preprocessed(data, "orig_train")

        # Perform differencing
        differenced_data = data.diff().dropna()
        
        # Store the last value of the original data before differencing
        last_value = data.iloc[-1]

        combined_data = combined_data.iloc[1:]  # Adjust the index after diff()
        combined_data.loc[:, "Value"] = differenced_data

        # Fit scaler and transform the data
        scaler = MinMaxScaler(feature_range=(-1, 1))
        combined_values = combined_data["Value"].values.reshape(-1, 1)
        combined_data["Value"] = scaler.fit_transform(combined_values)

        # Save the fitted scaler and the last value
        with open("data/train_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open("data/last_value.pkl", "wb") as f:
            pickle.dump(last_value, f)

        plot_preprocessed(combined_data["Value"], "train")

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

        last_series = df_long["Series"].unique()[-1]
        df_filtered = df_long[df_long["Series"] == last_series]
        plot_preprocessed(df_filtered["Value"], "orig_test")

        data = df_filtered["Value"]

        # Perform differencing starting from the last value of the training data
        data = pd.concat([pd.Series([last_value]), data])
        differenced_data = data.diff().dropna()
        
        combined_data = df_filtered.copy()
        combined_data.loc[:, "Value"] = differenced_data

        # Transform the differenced data using the scaler fitted on the training data
        combined_values = combined_data["Value"].values.reshape(-1, 1)
        combined_data["Value"] = scaler.transform(combined_values)

        plot_preprocessed(combined_data["Value"], "test")

    return combined_data
