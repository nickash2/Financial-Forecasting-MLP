# Description: This file contains the code to make predictions using the trained model.
import torch
from .mlp import SMAPELoss, MLP
import numpy as np
from sklearn.linear_model import LinearRegression


class Predictor:
    def __init__(self, model_path, device):
        self.model = MLP()  # Initialize the model first
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
            self.model.to(device)
        else:
            self.model.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )
        self.device = device

    def retrend_data(detrended_data, original_data):
        # Fit the linear model to the original data
        X = np.arange(len(original_data)).reshape(-1, 1)
        y = original_data.values.reshape(-1, 1)
        linreg = LinearRegression()
        linreg.fit(X, y)

        # Calculate the fitted values (trend)
        fitted_values = linreg.predict(X)

        # Add the trend to the detrended data
        retrended_data = detrended_data + fitted_values.flatten()

        return retrended_data

    @staticmethod
    def accuracy(true_data, predictions):
        numerator = np.abs(true_data - predictions)
        denominator = (np.abs(true_data) + np.abs(predictions)) / 2
        return np.mean(numerator / denominator) * 100

    def predict(self, data):
        # make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(data)
        return predictions

    def predict_next(self, last_window):
        self.model.eval()
        with torch.no_grad():
            # Convert the last window to a tensor, add a batch dimension, and move it to the device
            last_window = (
                torch.tensor(last_window, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            # Predict the next point
            prediction = self.model(last_window)
        return prediction.squeeze().cpu().numpy()
