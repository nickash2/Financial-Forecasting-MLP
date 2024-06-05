# This file contains the training loop for the MLP model
import time
from .mlp import MLP, SMAPELoss
import optuna
import torch
import numpy as np


INPUT_SIZE = 5 # window size, can be adjusted to any value if needed
OUTPUT_SIZE = 1 # next point

def evaluate(model, validation_loader, criterion):
    model.eval()  # set the model to evaluation mode

    validation_loss = 0.0

    with torch.no_grad():  # disable gradient computation
        for data, target in validation_loader:
            outputs = model(data)
            loss = criterion(outputs, target)
            validation_loss += loss.item()

    validation_loss /= len(validation_loader)
    return validation_loss

def train_epoch(model, train_loader, criterion, optimizer, device):
    model = model.to(device)
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


def objective(trial, train_loader, val_loader, device):
    # Define hyperparameters using trial.suggest_*
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    hidden_size = trial.suggest_int("hidden_size", 2, 5) 

    # Define your model and optimizer with the hyperparameters
    model = MLP(input_size=INPUT_SIZE, hidden_size=hidden_size, output_size=OUTPUT_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Define your loss function
    criterion = SMAPELoss()

    # Move model and criterion to the device
    model = model.to(device)
    criterion = criterion.to(device)

    # Train the model
    train_epoch(model, train_loader, criterion, optimizer, device)

    # Evaluate the model on your validation set
    validation_loss = evaluate(model, val_loader, criterion)

    return validation_loss

# ASK IF WE SHOULD JUST HAVE 80 20 SPLIT OF THE WHOLE DATASET FOR TESTING