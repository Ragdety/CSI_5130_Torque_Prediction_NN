import torch
import torch.nn as nn
import numpy as np
import os
import logging

from data.csc_data import CSCDataset
from data.df_preprocessing import (
  MinMaxScalerDP, CompositeDP, SequencesDP, 
  LowPassFilterDP, FeatureAdderDP, FeatureRemoverDP
)
from training import evaluate, train
from models import TorquePredictorFF, TorquePredictorLSTM
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.df_postprocessing import moving_average
from functions.constants import (
  FEATURES,
  BATCH_SIZE,
  N_EPOCHS,
  TRAIN_PERCENTAGE,
  LEARNING_RATE,
  MODEL_NAME,
  MODEL_PATH,
  MODEL_ARCH,
  IS_LSTM,
  MIN_MAX_SCALER_TYPE,
  TORQUE_LP_FILTER,
  USE_SCHEDULER,
  ADD_FEATURES
)
from scipy.signal import savgol_filter


def savitzky_golay_filter(data, window_size=5, polynomial_order=3):
    """Applies a Savitzky-Golay filter to the data."""
    return savgol_filter(data, window_size, polynomial_order)

def differentiate(data, time):
    """Differentiates data with respect to time."""
    return np.gradient(data, time)

def add_steering_velocity(df):
    """Adds a new feature to the dataframe: steering velocity."""
    
    df = df.copy()
    
   #  assert (df['t'].diff() > 0).all(), "Timestamps are not strictly increasing!"

    # df.loc[:, 'steeringVelocity'] = differentiate(df['steeringAngleDeg'], df['t'])
    df['steeringVelocity'] = df['steeringAngleDeg'].diff() / (df['t'].diff() + 1e-8)

    # TODO: Add/explore preprocessing for steering velocity
    # Replace inf/-inf with the maximum/minimum finite values
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    max_finite = df['steeringVelocity'][np.isfinite(df['steeringVelocity'])].max()
    min_finite = df['steeringVelocity'][np.isfinite(df['steeringVelocity'])].min()
    df['steeringVelocity'] = np.clip(df['steeringVelocity'], min_finite, max_finite)

    # df['steeringVelocity'] = df['steeringVelocity'].rolling(window=5, min_periods=1).mean()
    df.fillna(0, inplace=True)

    # Apply filter to remove noise
    df['steeringVelocity'] = savitzky_golay_filter(df['steeringVelocity'], 5, 2)
    # df['steeringVelocity'] = moving_average(df['steeringVelocity'], 1)

    return df

def add_steering_acceleration(df):
    """Adds a new feature to the dataframe: steering acceleration."""

    df = df.copy()

    # assert (df['t'].diff() > 0).all(), "Timestamps are not strictly increasing!"

    # df.loc[:, 'steeringAcceleration'] = differentiate(df['steeringVelocity'], df['t'])
    df['steeringAcceleration'] = df['steeringVelocity'].diff() / (df['t'].diff() + 1e-8)

    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    max_finite = df['steeringAcceleration'][np.isfinite(df['steeringAcceleration'])].max()
    min_finite = df['steeringVelocity'][np.isfinite(df['steeringAcceleration'])].min()
    df['steeringAcceleration'] = np.clip(df['steeringAcceleration'], min_finite, max_finite)

    # df['steeringAcceleration'] = df['steeringAcceleration'].rolling(window=5, min_periods=1).mean()
    df.fillna(0, inplace=True)

    # Apply filter to remove noise
    df['steeringAcceleration'] = savitzky_golay_filter(df['steeringAcceleration'], 5, 2)
    # df['steeringVelocity'] = moving_average(df['steeringVelocity'], 1)

    return df


def load_CSC_dataset(dataset_name='AUDI_Q3_2ND_GEN', is_prediction=False, pp_folder=None) -> CSCDataset:
    if is_prediction and not pp_folder:
        raise ValueError("Preprocessor path must be provided for prediction")
    
    save_path = None

    # TODO: get name automatically from preprocessor type
    # TODO: Add support for LSTM prediction
    if is_prediction:
        save_path = os.path.join(pp_folder, f"{MIN_MAX_SCALER_TYPE}_train_preprocessor_2.pkl")
        if IS_LSTM:
            print("Using LSTM preprocessor")
            save_path = os.path.join(pp_folder, f"{MIN_MAX_SCALER_TYPE}_train_preprocessor_0.pkl")
    
    train_preprocessor = MinMaxScalerDP(save_path=save_path)
    label_preprocessor = None

    # TODO: Improve preprocessing pipeline, instead of using a flag for each model
    # Might get very messy with more models being added
    if TORQUE_LP_FILTER:
        label_preprocessor = LowPassFilterDP()

    if ADD_FEATURES:
        old_train_pp = train_preprocessor
        train_preprocessor = CompositeDP([
            FeatureAdderDP([
                add_steering_velocity,
                add_steering_acceleration
            ]),
            FeatureRemoverDP(['t']),
            old_train_pp
        ])

    if IS_LSTM:
        print("Initializing LSTM preprocessor")
        train_preprocessor = CompositeDP([
            MinMaxScalerDP(save_path=save_path),
            SequencesDP()
        ])
        label_preprocessor = SequencesDP(is_label=True)

        if TORQUE_LP_FILTER:
            # Adding a low pass filter to torque for training
            label_preprocessor = CompositeDP([
                LowPassFilterDP(),
                SequencesDP(is_label=True)
            ])

    dataset = CSCDataset(dataset_name, 
                         FEATURES, 
                         download=True,
                         train_preprocessor=train_preprocessor,
                         label_preprocessor=label_preprocessor,
                         is_prediction=is_prediction,
                         logging_level=logging.INFO)
    total_dataset_size, num_csv = dataset.get_csv_metadata()

    print(f"Total file size of dataset: {round(total_dataset_size, 3)} MB")
    print(f"Number of CSV files in dataset: {num_csv}")

    return dataset


def split_dataset(dataset, train_percentage=TRAIN_PERCENTAGE):
    train_size = int(train_percentage * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def prepare_data_loaders(train_dataset: CSCDataset, val_dataset: CSCDataset, batch_size=BATCH_SIZE):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def initialize_model(input_features, device, out_activation=nn.Tanh):
    model = TorquePredictorFF(input_features, hidden_layers_arch=MODEL_ARCH, output_activation=out_activation)

    if IS_LSTM:
        model = TorquePredictorLSTM(input_features)

    model = model.to(device)

    print(f"Using device: {device}")

    return model


def train_model(model, train_loader, val_loader, device, dataset, use_scheduler=USE_SCHEDULER):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    criterion = torch.nn.MSELoss()
    
    scheduler = scheduler if use_scheduler else None

    return train(
        model, train_loader, val_loader, optimizer, criterion, N_EPOCHS, 
        device, MODEL_NAME, MODEL_PATH, dataset, scheduler=scheduler, log_interval=32
    )


def evaluate_model(model, val_loader, device):
    criterion = torch.nn.MSELoss()
    val_loss, predictions, actuals = evaluate(model, val_loader, criterion, device)

    # print(f"Validation loss: {val_loss}")

    return val_loss, predictions, actuals