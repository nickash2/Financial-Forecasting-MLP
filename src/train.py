# This file contains the training loop for the MLP model
from .mlp import MLP
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
            outputs = outputs.view(-1)
            loss = criterion(outputs, target)
            validation_loss += loss.item()

    validation_loss /= len(validation_loader)
    return validation_loss


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    val_loader,
    trial,
    epoch,
    num_epochs,
    early_stopping_patience=10
):
    model = model.to(device)
    model.train()
    total_loss = 0

    progress_bar = tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", position=0, leave=True
    )

    for inputs, targets in progress_bar:
        inputs = inputs.to(device)
         # Reshape the targets tensor to match the output tensor
        targets = targets.to(device).view(-1) 
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    val_loss = evaluate(model, val_loader, criterion, device)
    progress_bar.set_postfix(
        {"training_loss": "{:.3f}".format(avg_loss), "val_loss": val_loss}
    )

    trial.report(val_loss, epoch)

    if trial is not None and trial.should_prune():
        raise optuna.TrialPruned()

    print(
        f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}, Validation Loss: {val_loss}"
    )

    return avg_loss, val_loss


def train_final_model(train_loader, best_params, device):
    window_size = int(best_params["window_size"])
    hidden_size = int(best_params["hidden_size"])
    num_layers = int(best_params["hidden_layers"])
    num_epochs = int(best_params["num_epochs"])

    
    # print(f"Training final model with window_size={window_size}, hidden_size={hidden_size}, num_layers={num_layers}, num_epochs={num_epochs}")
    
    model = MLP(
        input_size=window_size,  # Use the window_size as the input_size
        hidden_size=hidden_size,
        output_size=OUTPUT_SIZE,
        num_layers=num_layers
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=best_params["lr"], weight_decay=best_params["lambda_reg"]
    )
    
    criterion = torch.nn.L1Loss()
    model = model.to(device)
    criterion = criterion.to(device)

    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", position=0, leave=True
        )

        for inputs, targets in progress_bar:
            # print(f"Input shape: {inputs.shape}")  # Debug statement
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1)  # Reshape the targets tensor to match the output tensor
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        progress_bar.set_postfix(
            {"training_loss": "{:.3f}".format(avg_loss)}
        )

        train_losses.append(avg_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}"
        )

    torch.save(model.state_dict(), "models/final_model.pth")

    # Plot training losses
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.savefig("plots/final_training_loss.png")





def objective(trial, dataset, device, n_splits=5, early_stopping_patience=10):
    blocked_split = BlockedTimeSeriesSplit(n_splits=n_splits)
    val_losses = []

    # Define hyperparameters using trial.suggest_*
    learning_rate = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    hidden_size = trial.suggest_categorical(
        "hidden_size", [2**i for i in range(4, 7)]
    )
    lambda_reg = trial.suggest_float(
        "lambda_reg", 1e-4, 1e-2, log=True
    )  # Increased upper limit
    hidden_layers = trial.suggest_int("hidden_layers", 1, 5)
    num_epochs = trial.suggest_int("num_epochs", 20, 40, step=5)
    window_size = trial.suggest_int("window_size", 2, 5)
    dataset.window_size = window_size

    for fold, (train_indices, val_indices) in enumerate(blocked_split.split(dataset)):
        train_subset = Subset(dataset, train_indices.tolist())
        val_subset = Subset(dataset, val_indices.tolist())
        train_loader = DataLoader(
            train_subset, batch_size=32, shuffle=False, drop_last=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=32, shuffle=False, drop_last=False
        )

        # Define model and optimizer with the hyperparameters
        model = MLP(
            input_size=window_size,
            hidden_size=hidden_size,
            output_size=OUTPUT_SIZE,
            num_layers=hidden_layers,
        ).to(device)
        
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=lambda_reg
        )

        # Define loss function
        criterion = torch.nn.L1Loss()

        # Train the model
        train_losses = []
        val_losses_fold = []
        best_val_loss = float('inf')
        early_stopping_counter = 0

        for epoch in range(num_epochs):
            train_loss, val_loss = train_epoch(model, train_loader, criterion, optimizer, device, val_loader, trial, epoch=epoch, num_epochs=num_epochs)  # type: ignore
            train_losses.append(train_loss)
            val_losses_fold.append(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_patience:
                print(f"\n- - - - -Early stopping triggered at epoch {epoch+1} - - - - -\n")
                break

        # Evaluate the model on the validation set
        val_losses.append(val_loss)

        # Check if the trial should be pruned after each fold
        if trial.should_prune():
            print(f"Trial {trial.number} pruned after fold {fold+1}")
            raise optuna.TrialPruned()

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
    plt.savefig(f"plots/trials/trial_{trial.number}_training_validation_loss.png")

    return avg_val_loss
