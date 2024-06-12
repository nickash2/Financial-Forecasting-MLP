# This file contains the implementation of the Multi-Layer Perceptron (MLP) model and the SMAPE loss function.
import torch
import torch.nn as nn


INPUT_SIZE = 5  # window size, can be adjusted to any value if needed
OUTPUT_SIZE = 1  # next point


class MLP(nn.Module):
    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=32,
        output_size=OUTPUT_SIZE,
        num_layers=2,
    ):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList()

        # Input layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_size))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = self.relu(hidden_layer(x))
        x = self.output_layer(x)
        return x


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
