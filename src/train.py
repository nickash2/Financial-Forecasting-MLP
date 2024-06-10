# This file contains the training loop for the MLP model
from .mlp import MLP, SMAPELoss
import optuna
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from .blockedcv import BlockedTimeSeriesSplit
import numpy as np

INPUT_SIZE = 5  # window size, can be adjusted to any value if needed
OUTPUT_SIZE = 1  # next point


def evaluate(model, validation_loader, criterion, device):
    model.eval()  # set the model to evaluation mode

    validation_loss = 0.0

    with torch.no_grad():  # disable gradient computation
        for data, target in validation_loader:
            data, target = (
                data.to(device),
                target.to(device),
            )  # Move data and target to the correct device
            outputs = model(data)
            loss = criterion(outputs, target)
            validation_loss += loss.item()

    validation_loss /= len(validation_loader)
    return validation_loss


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    # scheduler,  # Include scheduler
    device,
    val_loader,
    trial,
    epoch,
    num_epochs,
    early_stopping_patience=10,
):
    model = model.to(device)
    model.train()
    total_loss = 0
    early_stopping_counter = 0
    best_val_loss = float('inf')

    progress_bar = tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", position=0, leave=True
    )

    for inputs, targets in progress_bar:
        inputs = inputs.to(device)
        targets = targets.to(device)  # Reshape the targets tensor to match the output tensor
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    val_loss = evaluate(model, val_loader, criterion, device)
    # scheduler.step(val_loss)  # Update learning rate based on validation loss
    progress_bar.set_postfix(
        {"training_loss": "{:.3f}".format(avg_loss), "val_loss": val_loss}
    )

    trial.report(val_loss, epoch)
    if trial is not None:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1


        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered")
            return best_val_loss  # Report the best validation loss

    if trial is not None and trial.should_prune():
        raise optuna.TrialPruned()

    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}, Validation Loss: {val_loss}")
    return avg_loss, val_loss


def train_final_model(train_val_data, combined_train_val_loader, best_params, device):
    model = MLP(
        input_size=INPUT_SIZE,
        hidden_size=int(best_params["hidden_size"]),
        output_size=OUTPUT_SIZE,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=best_params["lr"], weight_decay=best_params["lambda_reg"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
    criterion = SMAPELoss()
    model = model.to(device)
    criterion = criterion.to(device)
    num_epochs = 50

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss, val_loss = train_epoch(
            model,
            combined_train_val_loader,
            criterion,
            optimizer,
            scheduler,  # Include scheduler
            device,
            combined_train_val_loader,  # Use combined_train_val_loader for validation
            None,  # trial is not needed in final training
            epoch,
            num_epochs=num_epochs,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    torch.save(model.state_dict(), "models/final_model.pth")

    # Plot training and validation losses
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.savefig("plots/final_training_validation_loss.png")


def objective(trial, dataset, device, n_splits=5):
    blocked_split = BlockedTimeSeriesSplit(n_splits=n_splits)
    val_losses = []

    for train_indices, val_indices in blocked_split.split(dataset):
        # Define hyperparameters using trial.suggest_*
        learning_rate = trial.suggest_float("lr", 1e-7, 1e-1, log=True)
        hidden_size = trial.suggest_int("hidden_size", 2, 8)
        lambda_reg = trial.suggest_float("lambda_reg", 1e-7, 1.0, log=True)  # Increased upper limit

        train_subset = Subset(dataset, train_indices.tolist())
        val_subset = Subset(dataset, val_indices.tolist())
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=False, drop_last=False)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, drop_last=False)

        # Define model and optimizer with the hyperparameters
        model = MLP(input_size=INPUT_SIZE, hidden_size=hidden_size, output_size=OUTPUT_SIZE).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_reg)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)


        # Define loss function
        criterion = SMAPELoss().to(device)

        # Train the model
        num_epochs = 50
        train_losses = []
        val_losses_fold = []
        for epoch in range(num_epochs):
            train_loss, val_loss = train_epoch(model, train_loader, criterion, optimizer,device, val_loader, trial, epoch=epoch, num_epochs=num_epochs)
            train_losses.append(train_loss)
            val_losses_fold.append(val_loss)

        # Evaluate the model on the validation set
        val_losses.append(val_loss)

    # Calculate the average validation loss across folds
    avg_val_loss = np.mean(val_losses)

    # Plot training and validation losses for each trial
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses_fold, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Losses for Trial {trial.number}")
    plt.legend()
    plt.savefig(f"plots/trial_{trial.number}_training_validation_loss.png")

    return avg_val_loss

