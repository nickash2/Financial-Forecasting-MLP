# This file contains the training loop for the MLP model
import time
from .mlp import MLP, SMAPELoss
import optuna
import torch
import numpy as np

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
        for inputs, targets, _ in train_loader:
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
    hidden_size = trial.suggest_int("hidden_size", 50, 200)
    input_size = trial.suggest_int("input_size", 1, 64)
    output_size = trial.suggest_int("output_size", 1, 64)


    # Define your model and optimizer with the hyperparameters
    model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
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

