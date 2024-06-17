# Description: This file contains the code to make predictions using the trained model.
import torch
from .mlp import SMAPELoss, MLP
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle


class Predictor:
    def __init__(self, best_params, device):
        self.device = device
        self.model = torch.load("models/final_model.pth")

        # Move model to device (GPU if available)
        if torch.cuda.is_available():
            self.model.to(device)

        # Set model to evaluation mode
        self.model.eval()

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

    def undo_normalization(self, data, scaler):
        # Reshape data to 2D array if it's currently 1D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Perform inverse transformation
        inversed_data = scaler.inverse_transform(data)

        # If the original data was 1D, flatten the inversed data
        if data.shape[1] == 1:
            inversed_data = inversed_data.flatten()

        return inversed_data

    def retrend_data(self, differenced_data, original_data):
        retrended_data = differenced_data + original_data.shift(1)
        return retrended_data.dropna().flatten()
