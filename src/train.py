# This file contains the training loop for the MLP model
from .mlp import MLP, SMAPELoss
import optuna
import torch


INPUT_SIZE = 5  # window size, can be adjusted to any value if needed
OUTPUT_SIZE = 1  # next point


def evaluate(model, validation_loader, criterion, device):
    model.eval()  # set the model to evaluation mode

    validation_loss = 0.0

    with torch.no_grad():  # disable gradient computation
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)  # Move data and target to the correct device
            outputs = model(data)
            loss = criterion(outputs, target)
            validation_loss += loss.item()

    validation_loss /= len(validation_loader)
    return validation_loss


def train_epoch(model, train_loader, criterion, optimizer, device, val_loader, trial):
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

        val_loss = evaluate(model, val_loader, criterion, device)
        trial.report(val_loss, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {val_loss}")


def objective(trial, train_loader, val_loader, device):
    # Define hyperparameters using trial.suggest_*
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    hidden_size = trial.suggest_int("hidden_size", 2, 5)
    lambda_reg = trial.suggest_float("lambda_reg", 1e-5, 1e-1, log=True)

    # Define model and optimizer with the hyperparameters
    model = MLP(input_size=INPUT_SIZE, hidden_size=hidden_size, output_size=OUTPUT_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_reg)

    # Define loss function
    criterion = SMAPELoss().to(device)

    # Move model and criterion to the device
    model = model.to(device)
    criterion = criterion.to(device)

    # Train the model
    train_epoch(model, train_loader, criterion, optimizer, device, val_loader, trial)

    # Evaluate the model on your validation set
    validation_loss = evaluate(model, val_loader, criterion, device)

    return validation_loss


# ASK IF WE SHOULD JUST HAVE 80 20 SPLIT OF THE WHOLE DATASET FOR TESTING
