import torch
import os
import numpy as np

from typing import Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from utils import get_nn_type, get_model_save_name


def evaluate(model: torch.nn.Module,
             val_loader: DataLoader,
             criterion: torch.nn.Module,
             device: torch.device) -> float:
    """
    Evaluate a model 

    Args:
        model: model to evaluate
        val_loader: validation data loader
        criterion: loss function
        device: device to run the model on

    Returns:
        val_loss: validation loss
        predictions: model predictions
        actuals: actual values
    """
    model.eval()
    val_loss = 0

    predictions = []
    actuals = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.squeeze(-1)

            # print("Output shape: ", output.shape)
            # print("Target shape: ", target.shape)

            predictions.append(output.cpu().numpy())
            actuals.append(target.cpu().numpy())

            val_loss += criterion(output, target).item()

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    val_loss /= len(val_loader)
    return val_loss, predictions, actuals



# Add types to the function signature
def train(model: torch.nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          n_epochs: int,
          device: torch.device,
          model_name: str,
          model_path: str,
          dataset: str,
          scheduler: ReduceLROnPlateau = None,
          log_interval: int = 100) -> Tuple[torch.nn.Module, list, list, str]:
    """
    Train a model

    Args:
        model: model to train
        train_loader: training data loader
        val_loader: validation data loader
        optimizer: optimizer
        criterion: loss function
        n_epochs: number of epochs
        device: device to run the model on
        model_name: name of the model
        model_path: path to save the model
        log_interval: number of batches to wait before logging the training status

    Returns:
        model: trained model
        train_losses: list of training losses
        val_losses: list of validation losses
        save_model: path to the saved model
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # data = data.view(data.size(0), data.size(1), -1)
            # print("Data shape: ", data.shape)
            output = model(data)
            output = output.squeeze(-1)

            # print("Batch idx: ", batch_idx)
            # print("Output shape: ", output.shape)
            # print("Target shape: ", target.shape)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                
        if scheduler:
            scheduler.step(train_loss)    

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print('\nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\n'.format(
            epoch, train_loss, val_loss))
        

        if val_loss < best_val_loss:
            nn_type = get_nn_type(model)
            save_model_name = get_model_save_name(model_name, nn_type, n_epochs, dataset)

            # Create folder with model name if it doesn't exist
            folder_name = save_model_name.split('.')[0]
            model_save_folder = os.path.join(model_path, folder_name)
            os.makedirs(model_save_folder, exist_ok=True)

            save_model = os.path.join(model_save_folder, save_model_name)

            best_val_loss = val_loss

            torch.save(model.state_dict(), save_model)
            print('Model saved at {}'.format(model_path))

    return model, train_losses, val_losses, save_model