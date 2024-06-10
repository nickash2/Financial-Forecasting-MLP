# This file contains the implementation of the Multi-Layer Perceptron (MLP) model and the SMAPE loss function.
import torch
import torch.nn as nn


INPUT_SIZE = 5  # window size, can be adjusted to any value if needed
OUTPUT_SIZE = 1  # next point

class MLP(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=4, output_size=OUTPUT_SIZE):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class SMAPELoss(nn.Module):
    def __init__(self):
        super(SMAPELoss, self).__init__()

    def forward(self, y_true, y_pred):
        # Ensure that the inputs are floats
        y_true = y_true.float()
        y_pred = y_pred.float()

        # Calculate the numerator and denominator
        numerator = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0

        # Avoid division by zero
        epsilon = 1e-8  # A small value to avoid division by zero
        smape = numerator / (denominator + epsilon)

        # Return the mean SMAPE multiplied by 100
        return 100 * torch.mean(smape)
