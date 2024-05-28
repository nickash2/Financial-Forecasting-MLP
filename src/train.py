# This file contains the training loop for the MLP model
import time
from mlp import MLP, SMAPELoss
import optuna
import torch


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


def train_epoch(model, train_loader, criterion, optimizer, model_name):
    model.train()

    running_loss = 0.0

    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # .backward() accumulates gradients

        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    end_time = time.time()

    running_loss /= len(train_loader)
    print("Training Loss: ", running_loss, "Time: ", end_time - start_time, "s")

    torch.save(model.state_dict(), f"../models/{model_name}.pth")
    return running_loss
