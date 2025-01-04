import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler
from scipy import signal
from functions.constants import (
  SEQUENCE_LENGTH, MIN_MAX_SCALER_TYPE, 
  COMPOSITE_TYPE, SEQUENCES_TYPE,
  ADDER_TYPE, REMOVER_TYPE
)


def low_pass_filter(data, wn=0.1, order=5):
    """Applies a low pass filter to the data."""

    b, a = signal.butter(order, wn, 'low', analog=False)
    return signal.filtfilt(b, a, data)


class SaveableDP:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.is_loaded = False

    def save(self, save_override: str = None):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
    
    def _check_save_path(self):
        if not self.save_path:
            raise ValueError("Save path for this preprocessor is not defined")


class MinMaxScalerDP(SaveableDP):
    def __init__(self, save_path=None, feature_range = (0, 1)):
        super(MinMaxScalerDP, self).__init__(save_path=save_path)
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.type = MIN_MAX_SCALER_TYPE
        self.is_loaded = False

    # Do the preprocessing on the actual np array data
    def __call__(self, df: pd.DataFrame):
        if self.is_loaded:
            return self.scaler.transform(df.values)
        
        # print(df.columns)
        data = df.values
        normalized_data = self.scaler.fit_transform(data)

        # print("Data: ", data)

        # print("Normalized Data: ", normalized_data)

        # print("Max: ", np.max(normalized_data))
        # print("Min: ", np.min(normalized_data))

        return normalized_data

        # return self.scaler.fit_transform(df.values)
    
    def save(self, save_override: str = None):
        self._check_save_path()
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.scaler, f)

    def load(self):
        self._check_save_path()

        with open(self.save_path, 'rb') as f:
            self.scaler = pickle.load(f)

        self.is_loaded = True

        
class SequencesDP:
    def __init__(self, sequence_length: int = SEQUENCE_LENGTH, is_label: bool = False):
        self.sequence_length = sequence_length
        self.is_label = is_label
        self.type = SEQUENCES_TYPE

    def __call__(self, df: pd.DataFrame):
        seq = []
        for i in range(len(df.values) - self.sequence_length):
            window = df.values[i:i+self.sequence_length]

            if self.is_label:
                seq.append(df.values[i+self.sequence_length])
            else:
                # seq.append(window)
                seq.append(df.values[i+self.sequence_length])

        # print(np.array(seq)[-1].shape)
        # return np.array(seq)[-1]
        return np.array(seq)
    
class LowPassFilterDP:
    def __init__(self, wn=0.1, order=2):
        self.wn = wn
        self.order = order

    def __call__(self, df: pd.DataFrame):
        filtered_data = low_pass_filter(df.values, wn=self.wn, order=self.order)
        return filtered_data.copy()

class FeatureAdderDP:
    def __init__(self, feature_funcs: list):
        self.feature_funcs = feature_funcs
        self.type = ADDER_TYPE

    def __call__(self, df: pd.DataFrame):
        for func in self.feature_funcs:
            df = func(df)
        return df
    
class FeatureRemoverDP:
    def __init__(self, features: list):
        self.features = features
        self.type = REMOVER_TYPE

    def __call__(self, df: pd.DataFrame):
        return df.drop(columns=self.features)
    
class CompositeDP(SaveableDP):
    def __init__(self, preprocessors: list, save_path=None):
        super(CompositeDP, self).__init__(save_path=save_path)
        self.preprocessors = preprocessors
        self.type = COMPOSITE_TYPE

    def __call__(self, data: pd.DataFrame):
        for preprocessor in self.preprocessors:
            data = preprocessor(data)
            # print data column names from dataframe with preprocessor name
            # print(f"Preprocessor: {preprocessor.type}; Data: {data.shape}")

        # print(data)
        return data

    def save(self):
        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, SaveableDP):
                preprocessor.save()
    
    def load(self):
        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, SaveableDP):
                preprocessor.load()

        self.is_loaded = True
