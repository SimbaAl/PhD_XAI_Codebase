"""
Data loading and saving utilities.
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
from scipy.io import loadmat
from .paths import PathManager


class DataLoader:
    """
    Handles loading and processing of data files.
    """

    @staticmethod
    def load_training_data(
            mobility: str,
            channel_model: str,
            modulation: str,
            scheme: str,
            snr: Union[int, str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load training dataset.

        Returns:
            Tuple of (inputs, targets) as PyTorch tensors
        """
        file_path = PathManager.get_training_data_path(
            mobility, channel_model, modulation, scheme, snr
        )

        with h5py.File(file_path, 'r') as file:
            dataset = file['DNN_Datasets']
            X = np.array(dataset['Train_X'])
            Y = np.array(dataset['Train_Y'])

        # Transpose if necessary (depending on how data is stored)
        X = np.transpose(X, (1, 0))
        Y = np.transpose(Y, (1, 0))

        return torch.from_numpy(X).float(), torch.from_numpy(Y).float()

    @staticmethod
    def load_testing_data(
            mobility: str,
            channel_model: str,
            modulation: str,
            scheme: str,
            snr: Union[int, str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load testing dataset."""
        file_path = PathManager.get_testing_data_path(
            mobility, channel_model, modulation, scheme, snr
        )

        # Use loadmat instead of h5py
        data = loadmat(str(file_path))
        dataset = data['DNN_Datasets']

        # Access the data correctly based on the MATLAB structure
        X = dataset['Test_X'][0, 0]
        Y = dataset['Test_Y'][0, 0]

        return torch.from_numpy(X).float(), torch.from_numpy(Y).float()

    @staticmethod
    def save_processed_data(
            data: np.ndarray,
            mobility: str,
            channel_model: str,
            modulation: str,
            scheme: str,
            data_type: str,
            snr: Union[int, str]
    ) -> None:
        """Save processed data (e.g., relevance scores)."""
        file_path = PathManager.get_processed_data_path(
            mobility, channel_model, modulation, scheme, data_type, snr
        )
        np.save(file_path, data)

    @staticmethod
    def load_processed_data(
            mobility: str,
            channel_model: str,
            modulation: str,
            scheme: str,
            data_type: str,
            snr: Union[int, str]
    ) -> np.ndarray:
        """Load processed data."""
        file_path = PathManager.get_processed_data_path(
            mobility, channel_model, modulation, scheme, data_type, snr
        )
        return np.load(file_path)