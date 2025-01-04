import pandas as pd
import os
import torch
import glob
import logging

from torch.utils.data import Dataset
from typing import Any, List, Optional, Callable
from utils import download_and_extract, move_csv_files, pad_dataframe
from data.df_preprocessing import SaveableDP, FeatureAdderDP


# CSC = Comma Steering Control
# Defining a custom dataset here to use it in DataLoader later

# Generic CSC base class to specify 
# dataset name and download it automatically, unzip it, extract
# features (vehicle dynamics) and label (torque)
class CSCDataset(Dataset):
    base_url = "https://huggingface.co/datasets/commaai/commaSteeringControl/resolve/main/data/"
    storage_dir = os.path.join(os.getcwd(), 'storage')
    
    def __init__(self, 
                 dataset: str, 
                 features: List, 
                 label: str = 'steerFiltered',
                 download: bool = False,
                 train_preprocessor: Optional[Callable] = None,
                 label_preprocessor: Optional[Callable] = None,
                 is_prediction: bool = False,
                 logging_level: int = logging.INFO,) -> None:
        """
        Defines a Comma Steering Control (CSC) dataset for a specific car model
        <https://huggingface.co/datasets/commaai/commaSteeringControl/tree/main/data>

        Args:
            dataset: dataset to download from hugging face (Ex: acura_rdx_2020.zip or acura_rdx_2020)
            feature_columns: CSC feature colums of your choosing
            label_column: CSC label column. steerFiltered by default (torque)
            download: Boolean to specify if it should download when instantiating this class
        """

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

        dataset_zip = self._preproc_dataset_name(dataset)

        # Folder name is the same as the zip file name without the .zip
        dataset_folder_name = dataset_zip.split('.')[0]

        # Dataset props
        self.dataset_zip = dataset_zip
        self.url = self.base_url + dataset_zip
        self.dataset_folder_path = os.path.join(self.storage_dir, dataset_folder_name)
        
        if download:
            self.download()
        else:
            self._check_integrity()

        # Training features props
        self.features = features
        self.label = label
        self.dataframes = []
        self.train_preprocessor = train_preprocessor
        self.label_preprocessor = label_preprocessor
        self.is_prediction = is_prediction

        # 600 as a baseline, will be updated below
        # max_rows is the maximum number of rows a csv file has
        # This is to make sure the dataframe can be padded with 0's for
        # Torch's dataloader to work
        self.max_rows = 600

        for file in os.listdir(self.dataset_folder_path):
            csv_file_path = os.path.join(self.dataset_folder_path, file)
            df = pd.read_csv(csv_file_path)
            self.dataframes.append(df)
            
            if len(df) > self.max_rows:
                self.max_rows = len(df) 

    def __len__(self) -> int:
        _, num_csv = self.get_csv_metadata()
        return num_csv
    
    def __getitem__(self, idx) -> Any:
        # Select the dataframe (csv) to use
        df = self.dataframes[idx]
        df = pad_dataframe(df, max_rows=self.max_rows)

        features = df[self.features]
        label = df[self.label]

        # Preprocess the features dataframe
        if self.train_preprocessor:
            if self.is_prediction and isinstance(self.train_preprocessor, SaveableDP):
                self.train_preprocessor.load()
            features = self.train_preprocessor(features)

        if self.label_preprocessor:
            if self.is_prediction and isinstance(self.label_preprocessor, SaveableDP):
                self.label_preprocessor.load()
            label = self.label_preprocessor(label)
            # print(label)

        # Convert to torch tensors
        features = torch.tensor(features, dtype=torch.float64)
        label = torch.tensor(label, dtype=torch.float64)

        self.logger.debug(f"Features shape: {features.shape}, Label shape: {label.shape}")
        
        return features, label

    def download(self) -> None:
        if self._is_downloaded_extracted():
            self.logger.info(f"Dataset is already downloaded and extracted and ready to use! Folder: {self.dataset_folder_path}")
            return

        download_and_extract(self.url, self.storage_dir)

        # Check if the folder is repeated (meaning there are 2 folders with same name)
        # Ex: FolderName/FolderName/example.csv
        dataset_folder_name = os.path.basename(self.dataset_folder_path)
        double_folder = os.path.join(self.dataset_folder_path, dataset_folder_name)
        
        if os.path.exists(double_folder):
            move_csv_files(double_folder, self.dataset_folder_path)
            os.removedirs(double_folder)

        zip = os.path.join(self.storage_dir, self.dataset_zip)
        os.remove(zip)
        
        self._check_integrity()

    def get_csv_metadata(self):
        """
        Gets metadata of the total number of csv's

        Returns (Tuple):
            total_size (in MB), number_of_csv_files
        """
        csv_files = glob.glob(f'{self.dataset_folder_path}/*.csv')
        total_size = 0
        csv_count = 0
        
        for file in csv_files:
            total_size += os.path.getsize(file)
            csv_count += 1

        size_mb = total_size / (1024 * 1024)
        
        return size_mb, csv_count

    def _check_integrity(self) -> None:
        # Might add more logic in this function (Ex: checking the download is not corrupted, etc..)
        if not self._is_downloaded_extracted():
            raise Exception(f"Dataset {self.dataset_zip} not downloaded/extracted...")
            
        self.logger.info(f"Dataset downloaded and extracted, ready to use! Folder: {self.dataset_folder_path}")
    
    def _is_downloaded_extracted(self) -> bool:
        if not os.path.exists(self.dataset_folder_path):
            return False

        # Return True if folder is not empty 
        return not len(os.listdir(self.dataset_folder_path)) == 0

    def _preproc_dataset_name(self, dataset_name: str) -> str:
        """
        Preprocess dataset name to add .zip if it doesn't have it
        """

        add_zip = dataset_name + '.zip'
        return add_zip if '.zip' not in dataset_name else dataset_name
    