import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
import numpy as np
import pickle
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def preprocess(dataset, test=False):
    df_long = pd.melt(
        dataset,
        id_vars=["Series", "N", "NF", "Category", "Starting Year", "Starting Month"],
        var_name="Month",
        value_name="Value",
    )

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

    data = df_long["Value"]

    X = np.arange(len(data)).reshape(-1, 1)
    y = data.values.reshape(-1, 1)

    if not test:
        lasso.fit(X, y)

    fitted_values = lasso.predict(X)
    detrended_data = data - fitted_values.flatten()
    df_long.loc[:, "Value"] = detrended_data

    if not test:
        df_long.loc[:, "Value"] = scaler.fit_transform(
            df_long["Value"].values.reshape(-1, 1)
        )
    else:
        df_long.loc[:, "Value"] = scaler.transform(
            df_long["Value"].values.reshape(-1, 1)
        )

    df_final = df_long

    if not test:
        with open("data/linear_regr.pkl", "wb") as f:
            pickle.dump(lasso, f)
        with open("data/train_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

    return df_final


from sklearn.metrics import mean_absolute_error

def objective(trial, X_train, y_train, X_test, y_test):
    alpha = trial.suggest_float('alpha', 1e-4, 1e2, log=True)
    max_iter = trial.suggest_int('max_iter', 1000, 10000)
    lasso = Lasso(alpha=alpha, max_iter=max_iter)

    # Fit the model on the training data
    lasso.fit(X_train, y_train)

    # Predict the validation data
    fitted_values = lasso.predict(X_test)

    # Calculate the mean absolute error on the validation data
    mae = mean_absolute_error(y_test, fitted_values)

    return mae

# if __name__ == "__main__":
#     df = pd.read_excel("data/M3C.xls", sheet_name="M3Month")
#     df.dropna(axis=1, inplace=True)

#     df_long = pd.melt(
#         df,
#         id_vars=["Series", "N", "NF", "Category", "Starting Year", "Starting Month"],
#         var_name="Month",
#         value_name="Value",
#     )

#     X = np.arange(len(df_long["Value"])).reshape(-1, 1)
#     y = df_long["Value"].values.reshape(-1, 1)

#     # Split the data into training and validation sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=122, shuffle=True)

#     study = optuna.create_study(direction='minimize')
#     study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=200)

#     best_params = study.best_params
#     print(f"Best parameters: {best_params}")

#     lasso = Lasso(alpha=best_params['alpha'], max_iter=best_params['max_iter'])

#     lasso.fit(X, y)

#     with open("data/linear_regr.pkl", "wb") as f:
#         pickle.dump(lasso, f)

#     plt.plot(X, y, label='True Data')
#     plt.plot(X, lasso.predict(X), label='Fitted Data')
#     plt.legend()
#     plt.show()