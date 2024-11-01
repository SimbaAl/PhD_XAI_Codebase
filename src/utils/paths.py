"""
Path management utilities for the project.
"""

from pathlib import Path
from typing import Union, Dict


# Define project root and major directories as class attributes
class PathManager:
    """
    Manages paths for data, models, and results.
    """
    # Define base directories
    PROJECT_ROOT = Path("/home/aldrin/turmite/LSTM/TCN/2_channel_TCN/maybe/PhD_Codebase")
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"

    @classmethod
    def get_model_path(
            cls,
            mobility: str,
            channel_model: str,
            modulation: str,
            scheme: str,
            snr: Union[int, str]
    ) -> Path:
        """Get path for a trained model."""
        filename = f"{mobility}_{channel_model}_{modulation}_{scheme}_DNN_{snr}.pt"
        return cls.MODELS_DIR / "trained" / filename

    @classmethod
    def get_training_data_path(
            cls,
            mobility: str,
            channel_model: str,
            modulation: str,
            scheme: str,
            snr: Union[int, str]
    ) -> Path:
        """Get path for training dataset."""
        filename = f"{mobility}_{channel_model}_{modulation}_{scheme}_DNN_training_dataset_{snr}.mat"
        return cls.DATA_DIR / "raw" / filename

    @classmethod
    def get_testing_data_path(
            cls,
            mobility: str,
            channel_model: str,
            modulation: str,
            scheme: str,
            snr: Union[int, str]
    ) -> Path:
        """Get path for testing dataset."""
        filename = f"{mobility}_{channel_model}_{modulation}_{scheme}_DNN_testing_dataset_{snr}.mat"
        return cls.DATA_DIR / "raw" / filename

    @classmethod
    def get_processed_data_path(
            cls,
            mobility: str,
            channel_model: str,
            modulation: str,
            scheme: str,
            data_type: str,
            snr: Union[int, str]
    ) -> Path:
        """Get path for processed data."""
        filename = f"{mobility}_{channel_model}_{modulation}_{scheme}_{data_type}_{snr}.npy"
        return cls.DATA_DIR / "processed" / filename

    @classmethod
    def get_results_path(
            cls,
            mobility: str,
            channel_model: str,
            modulation: str,
            scheme: str,
            result_type: str
    ) -> Path:
        """Get path for results."""
        base_dir = cls.RESULTS_DIR / f"{mobility}_{channel_model}_{modulation}_{scheme}"
        return base_dir / result_type

    @classmethod
    def create_required_directories(cls) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            cls.DATA_DIR / "raw",
            cls.DATA_DIR / "processed",
            cls.MODELS_DIR / "trained",
            cls.MODELS_DIR / "checkpoints",
            cls.RESULTS_DIR
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def verify_data_exists(
            cls,
            mobility: str,
            channel_model: str,
            modulation: str,
            scheme: str,
            training_snr: Union[int, str],
            test_snrs: list
    ) -> Dict[str, bool]:
        """Verify that all required data files exist."""
        files_exist = {}

        # Check model file
        model_path = cls.get_model_path(
            mobility, channel_model, modulation, scheme, training_snr
        )
        files_exist['model'] = model_path.exists()

        # Check training data
        train_path = cls.get_training_data_path(
            mobility, channel_model, modulation, scheme, training_snr
        )
        files_exist['training_data'] = train_path.exists()

        # Check testing data for all SNRs
        files_exist['testing_data'] = {}
        for snr in test_snrs:
            test_path = cls.get_testing_data_path(
                mobility, channel_model, modulation, scheme, snr
            )
            files_exist['testing_data'][snr] = test_path.exists()

        return files_exist