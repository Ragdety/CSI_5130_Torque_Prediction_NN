import os
import torch
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn

from torch.utils.data import DataLoader
from functions.main import (
    load_CSC_dataset,
    split_dataset,
    prepare_data_loaders,
    initialize_model,
    train_model,
    evaluate_model
)
from functions.constants import FEATURES, IS_LSTM, BATCH_SIZE, NUM_FEATURES, ADD_FEATURES
from data.df_postprocessing import low_pass_filter, moving_average
from prediction import predict_torque, load_model
from utils import safe_save_or_show_plot, save_preprocessor
from pathlib import Path


def plot_and_save_loss(train_losses, val_losses, model_name):
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()

    # Save the plot Train_Val_Loss_Graphs
    save_fig_path = os.path.join(os.getcwd(), f'saved_models/{model_name}', f'train_loss_graph.png')
    safe_save_or_show_plot(plt, save_fig_path)

def plot_and_save_predictions(actual_torque, 
                              predicted_torque, 
                              csv_id, 
                              model_path,
                              dataset):
    plt.figure(figsize=(10, 5))
    plt.plot(actual_torque.flatten(), label="Actual Torque")
    plt.plot(predicted_torque.flatten(), label="Predicted Torque")
    plt.xlabel("Time Step")
    plt.ylabel("Torque")
    plt.title("Actual vs Predicted Torque")
    plt.legend()

    # Save with template: csv_csv_id_dataset.png
    model_name = os.path.basename(model_path).split('.')[0]
    prediction_plot_name = f"csv_{csv_id}.png"

    # Create dataset folder if it doesn't exist
    save_folder_path = os.path.join('../prediction_images', model_name, dataset)
    os.makedirs(save_folder_path, exist_ok=True)

    save_fig_path = os.path.join(save_folder_path, prediction_plot_name)
    safe_save_or_show_plot(plt, save_fig_path)

def calculate_accuracy(y_pred, y_true):
    """Calculates accuracy given predicted and true labels."""

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.argmax(dim=1)

    correct = (y_pred == y_true).sum().item()
    total = y_true.size(0)
    accuracy = correct / total

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Train or predict with torque model")
    
    # Define --train and --predict as mutually exclusive options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train the model")
    group.add_argument("--predict", action="store_true", help="Predict torque values")

    # Additional arguments for each mode
    parser.add_argument("--dataset", type=str, help="Dataset name for training")
    parser.add_argument("--model", type=str, help="Path to the model for prediction")
    parser.add_argument("--csv", type=str, help="CSV file ID for prediction")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if args.train:
        # Check if dataset name is provided for training
        if not args.dataset:
            raise ValueError("You must specify a dataset name using --dataset for training.")

        # Load, split, and prepare the dataset for training
        dataset = load_CSC_dataset(args.dataset)
        train_dataset, val_dataset = split_dataset(dataset)
        train_loader, val_loader = prepare_data_loaders(train_dataset, val_dataset)

        # Initialize and train the model
        model = initialize_model(NUM_FEATURES, device)
        exit_code = None

        try:
            model, train_losses, val_losses, saved_model = train_model(model, 
                                                                      train_loader,
                                                                      val_loader, 
                                                                      device, 
                                                                      args.dataset)

            # Evaluate the model on the validation set
            val_loss, predictions, actuals = evaluate_model(model, val_loader, device)
            print(f"Validation loss: {val_loss}")
        except KeyboardInterrupt:
            # TODO: propagate try-except to train_model to return train_losses, val_losses, saved_model
            print("Training interrupted by user...")
            print("Still plotting training and validation loss...")
            exit_code = 1

        # Plot training and validation losses
        print("Plotting training and validation loss...")
        model_name = os.path.basename(saved_model).split('.')[0]
        plot_and_save_loss(train_losses, val_losses, model_name)

        # Move preprocessors to the correct folder
        saved_model_path = Path(saved_model)
        model_pp_folder = saved_model_path.parent.absolute() / 'preprocessors'
        os.makedirs(model_pp_folder, exist_ok=True)

        if dataset.train_preprocessor:
            save_preprocessor(model_pp_folder, dataset.train_preprocessor, is_label=False)
        
        if dataset.label_preprocessor:
            save_preprocessor(model_pp_folder, dataset.label_preprocessor, is_label=True)

        if exit_code:
            exit(exit_code)

    elif args.predict:
        # Check if model path is provided for prediction
        if not args.dataset:
            raise ValueError("You must specify a dataset name using --dataset for predictions.")
        if not args.model:
            raise ValueError("You must specify a model path using --model_path for prediction.")
        if not args.csv:
            raise ValueError("You must specify a CSV ID using --csv for prediction.")

        # Get base folder
        saved_model_path = Path(args.model)
        model_pp_folder = saved_model_path.parent.absolute() / 'preprocessors'        

        # Load the model and dataset for prediction
        prediction_dataset = load_CSC_dataset(args.dataset, is_prediction=True, pp_folder=model_pp_folder)
        prediction_dataloader = DataLoader(prediction_dataset, batch_size=1, shuffle=False)

        # Make predictions and plot actual vs. predicted torque
        predicted_torque = None
        actual_torque = None

        # CSV ID for prediction
        csv_id = int(args.csv)  

        # TODO: predict with lstm 750 epochs (BEST loss curves so far)
        for idx, (data, target) in enumerate(prediction_dataloader):
            if idx != csv_id:
                continue
            
            output_activation = nn.Tanh
            print(target.shape)

            if os.path.basename(args.model) == 'torque_predictor_ff_150_Epochs_AUDI_Q3_2ND_GEN.pt' :
                output_activation = nn.Sigmoid
            
            # TODO: Fix this hard coded way of getting the preprocessor
            if IS_LSTM:
                min_max_scaler = prediction_dataset.train_preprocessor.preprocessors[0].scaler
            else:
                if ADD_FEATURES:
                    min_max_scaler = prediction_dataset.train_preprocessor.preprocessors[2].scaler
                else:
                    min_max_scaler = prediction_dataset.train_preprocessor.scaler
            predicted_torque = predict_torque(args.model, data, FEATURES, output_activation, min_max_scaler)

            # Saturation of ndarray values > 2 
            # predicted_torque[predicted_torque > 2] = 2

            # predicted_torque += 4.75
            # TOyota *= 2

            # Other models:
            # predicted_torque += 0.25
            predicted_torque *= 1.8

            
            # Toyota trained model:
            # predicted_torque -= 0.05

            if IS_LSTM:
                predicted_torque /= 3

            print("Predicted Torque Shape", predicted_torque.shape)
            print("Target Shape", target.shape)

            actual_torque = target.squeeze() # * 3
            # actual_torque = low_pass_filter(actual_torque, 0.2, 3)
            break
        
        accuracy = calculate_accuracy(predicted_torque, actual_torque)
        print(f"Accuracy: {accuracy}")

        # Plot actual vs predicted torque
        plot_and_save_predictions(actual_torque, 
                                  predicted_torque, 
                                  csv_id, 
                                  args.model, 
                                  args.dataset)


if __name__ == "__main__":
    main()