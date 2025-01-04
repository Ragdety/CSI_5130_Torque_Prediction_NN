import numpy as np

from scipy import signal


def savitzky_golay_filter(data, window_size, polynomial_order):
    """Applies a Savitzky-Golay filter to the data."""
    
    return signal.savgol_filter(data, window_size, polynomial_order)

def low_pass_filter(data, wn=0.1, order=5):
    """Applies a low pass filter to the data."""

    b, a = signal.butter(order, wn, 'low', analog=False)
    return signal.filtfilt(b, a, data)

def moving_average(data, window_size=2):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')