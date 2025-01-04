import zipfile
import os
import shutil
import glob
import urllib
import urllib.error
import urllib.request
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from functions.constants import MAX_CSC_ROWS, USER_AGENT
from models import MODEL_TYPES
from data.df_preprocessing import SaveableDP, CompositeDP


# Function to pad a dataframe to have all tensors the same shape 
def pad_dataframe(df, max_rows=MAX_CSC_ROWS, padding_value=0):
    if len(df) < max_rows:
        # Padding is a dataframe of 0's
        padding = pd.DataFrame(padding_value, index=range(max_rows - len(df)), columns=df.columns)
        
        # Combine the padding df to the current df 
        return pd.concat([df, padding], ignore_index=True)
    
    return df


# From PyTorch Vision Utils:
def _urlretrieve(url, filename, chunk_size=1024 * 32) -> None:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
        with open(filename, "wb") as fh, tqdm(total=response.length) as pbar:
            while chunk := response.read(chunk_size):
                fh.write(chunk)
                pbar.update(len(chunk))

def download_from_url(
    url: str,
    download_dir: str,
    filename: str = None
):
    if not filename:
        filename = os.path.basename(url)
        
    os.makedirs(download_dir, exist_ok=True)
    
    file_path = os.fspath(os.path.join(download_dir, filename))

    if os.path.isfile(file_path):
        print(f"File already downloaded: {file_path}")
        return file_path

    print("This might take a while if file is too large or your internet connection is slow...")
    print(f"Downloading file: {url}")
    
    try:
        _urlretrieve(url, file_path)
    except (urllib.error.URLError, OSError) as e:
        print(f"There was an error while trying to download file from url: {url}")
        raise e

    # check integrity of downloaded file
    if not os.path.isfile(file_path):
        raise RuntimeError("File not found or corrupted.")

    print(f"Downloaded file successfully!")

    # Return downloaded file path for post-processing
    return file_path


def unpack_zip_file(dataset_zip, storage_dir):
    print(f"Extracting zip: {dataset_zip}...")

    zip_file_name = os.path.basename(dataset_zip)
    csv_folder_name = zip_file_name.split('.')[0]
    extract_to_dir = os.path.join(storage_dir, csv_folder_name)
    
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to_dir)
        print(zip_ref.namelist()[:5])

    print(f"File was extracted to: {extract_to_dir}")
    
    # Return the folder where files were extracted 
    return extract_to_dir

def move_csv_files(current, destination) -> int:
    csv_files = glob.glob(current + '/*.csv')
    destination_folder = os.path.join(destination)

    if os.listdir(current) == 0:
        print(f"Folder is empty: {current}")
        return 0

    csv_count = 0

    print("Moving CSV files...")
    
    os.makedirs(destination_folder, exist_ok=True)
    for file in csv_files:
        shutil.move(file, destination_folder)
        csv_count += 1

    print(f"Moved {csv_count} csv's to folder: {destination_folder}...")
    return csv_count

def download_and_extract(url, download_dir):
    dataset_zip = download_from_url(url, download_dir)
    return unpack_zip_file(dataset_zip, download_dir)

def get_nn_type(model: nn.Module) -> str:
    return MODEL_TYPES[type(model)]

# Can't assign dataset to CSCDataset because of circular imports
def get_CSC_dataset_name(dataset) -> str:
    return dataset.dataset_zip.split('.')[0]

def get_model_save_name(model_name: str, nn_type: str, n_epochs: int, dataset: str) -> str:
    return f'{model_name}_{nn_type}_{n_epochs}_Epochs_{dataset}.pt'

def safe_save_or_show_plot(plt, save_fig_path):
    """
    Saves the plot to the specified path. If the path is invalid, the plot is displayed instead.
    This allows the user to see the plot even if the path is incorrect, they can save it manually if needed

    :param plt: The plot to save
    :param save_fig_path: The path to save the plot
    """
    
    try:
        plt.savefig(save_fig_path)
    except Exception as e:
        print("ERROR: Unable to save the plot. Save manually if needed and check the path.")
        plt.show()
        raise e

    plt.show()

def save_preprocessor(save_folder: str, preprocessor: SaveableDP, is_label: bool = False) -> None:
    if not isinstance(preprocessor, SaveableDP):
        print("Preprocessor must be an instance of SaveableDP to save...")
        print("Preprocessor not saved.")
        return

    pp_type = preprocessor.type

    pp_ = "label" if is_label else "train"

    # TODO: Might add dictionary to keep an order of preprocessors

    if not isinstance(preprocessor, CompositeDP):
        preprocessor.save_path = f"{save_folder}/{pp_type}_{pp_}_preprocessor.pkl"
        preprocessor.save()
        print(f"Saved train preprocessor to {preprocessor.save_path}")
        return
    
    # Handle composite preprocessors
    for idx, pp in enumerate(preprocessor.preprocessors):
        if not isinstance(pp, SaveableDP):
            print(f"Preprocessor at index {idx} is not an instance of SaveableDP. Skipping...")
            continue
        pp_type = pp.type
        pp.save_path = f"{save_folder}/{pp_type}_{pp_}_preprocessor_{idx}.pkl"
        pp.save()
        print(f"Saved train preprocessor to {pp.save_path}")
    
