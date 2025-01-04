import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from models import TorquePredictorFF, TorquePredictorLSTM
from functions.constants import FEATURES, MODEL_ARCH, IS_LSTM, SEQUENCE_LENGTH, ADD_FEATURES
from sklearn.preprocessing import MinMaxScaler


def load_model(model_path, output_activation, device='cuda'):
    if IS_LSTM:
        model = TorquePredictorLSTM(len(FEATURES), return_sequences=True)
    else:
        if ADD_FEATURES:
            model = TorquePredictorFF(6, MODEL_ARCH, output_activation=output_activation)
        else:
            model = TorquePredictorFF(len(FEATURES), MODEL_ARCH, output_activation=output_activation)

    model.to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True,), strict=False)
    model.eval()

    return model

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        window = data[i:i+sequence_length]
        sequences.append(window)
    return np.array(sequences)
  

def preprocess_data(file_path, features=FEATURES):
    """
    Preprocesses new data for prediction
    :param file_path: Path to new data (csv file)
    :param features: Features used in training
    """
    df = pd.read_csv(file_path)
    df = df[features]

    scaler = MinMaxScaler(feature_range=(0, 1))
    values = scaler.fit_transform(df.values)

    # if IS_LSTM:
    #     # TODO: Toyota dataset has different sequence length = 25
    #     values = create_sequences(values, sequence_length=SEQUENCE_LENGTH)

    data = torch.tensor(values, dtype=torch.float64)
    return data.unsqueeze(0), scaler


def predict_torque(model_path, new_data, features, output_activation: nn.Module = nn.Tanh, scaler=None, device='cuda'):
    """
    Predicts torque values using the trained model

    :param model_path: Path to the trained model
    :param new_data: Can be a tensor or a path to a csv file
    :param features: Features used in training
    :param output_activation: Output activation function

    :return: Predicted torque values
    """
    model = load_model(model_path, output_activation)

    if isinstance(new_data, str):
      raise NotImplementedError("Preprocessing new data from a csv file is not implemented yet")
      data = preprocess_data(new_data, features)
    elif isinstance(new_data, torch.Tensor):
      data = new_data.to(device)
      if IS_LSTM:
          # data = data.squeeze(1)
          print(f"Data shape: {data.shape}")
      else:
          data = data.unsqueeze(0)
    else: 
      raise ValueError("new_data should be a path to a csv file or a tensor")

    with torch.no_grad():
        torque_predictions = model(data)
        print(f"Predictions shape: {torque_predictions.shape}")
        # print(f"Predictions: {torque_predictions}")

    predictions_np = torque_predictions.cpu().numpy()
    # predictions_np = predictions_np.squeeze()  # Shape: (601,)
    predictions_np = predictions_np.reshape(-1, 1)  # Shape: (601, 1)
    print(f"Predictions Numpy shape: {predictions_np.shape}")
    
     # Apply inverse transformation if scaler is provided
    if scaler is not None:
        inverse_predictions = predictions_np
        # Pad with dummy values to match scaler's expected shape
        # dummy_features = np.zeros((predictions_np.shape[0], scaler.n_features_in_ - 1))
        # padded_predictions = np.hstack((predictions_np, dummy_features))  # Shape: (601, 4)

        # # Apply inverse transformation
        # inverse_predictions = scaler.inverse_transform(padded_predictions)

        # # Extract the relevant column (torque values)
        # inverse_predictions = inverse_predictions[:, 0]# .reshape(-1, 1)  # Shape: (601, 1) 
        # print(f"Inverse predictions shape: {inverse_predictions.shape}")
        # print(f"Inverse predictions: {inverse_predictions}")
    else:
        inverse_predictions = predictions_np  # Return raw predictions if scaler is not provided


    return inverse_predictions