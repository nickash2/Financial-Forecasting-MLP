import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


def plot_preprocessed(df_detrended):
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

    plt.savefig("plots/preprocessed_series.png")


def preprocess():
    df = pd.read_excel("data/M3C.xls", sheet_name="M3Month")
    df.dropna(axis=1, inplace=True)

    # *As* the data is in wide format, we convert it to a long format.
    df_long = pd.melt(
        df,
        id_vars=["Series", "N", "NF", "Category", "Starting Year", "Starting Month"],
        var_name="Month",
        value_name="Value",
    )

    # To pick our group of interest, we decided to pick the "Micro", as it has a lot more data.

    df_monthly = df_long[df_long["Category"] == "MICRO       "]

    series_to_plot = df_monthly["Series"].unique()

    # Create an empty DataFrame to store the detrended and de-seasonalized data
    df_detrended = pd.DataFrame()

    for series in series_to_plot:
        # Filter the DataFrame for the current series
        df_filtered = df_monthly[df_monthly["Series"] == series]

        # Decompose the data with period of 12 months
        decomposition = sm.tsa.seasonal_decompose(
            df_filtered["Value"], model="additive", period=12
        )

        # Get the detrended and de-seasonalized data
        detrended_deseasonalized = decomposition.resid

        # Create a DataFrame for the current series
        df_current = df_filtered.copy()
        df_current["Value"] = detrended_deseasonalized

        # Append the current DataFrame to the main DataFrame
        df_detrended = pd.concat([df_detrended, df_current])

    # Now df_detrended contains the detrended and de-seasonalized data
    df_detrended.dropna(inplace=True)
    return df_detrended
