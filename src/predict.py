# Description: This file contains the code to make predictions using the trained model.
import torch
from .mlp import SMAPELoss


class Predictor:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()

    def re_trend(self, data, trend):
        pass

    def accuracy(self, data, predictions):
        pass

    def predict(self, data):
        # load the best hyperparameters

        # make predictions

        pass
