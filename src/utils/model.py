"""
Model loading and saving utilities.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
from src.models import DNN
from .paths import PathManager


class ModelManager:
    """
    Handles loading and saving of models.
    """

    @staticmethod
    def load_model(
            mobility: str,
            channel_model: str,
            modulation: str,
            scheme: str,
            snr: int,
            model_params: Dict[str, Any],
            device: Optional[torch.device] = None
    ) -> DNN:
        """
        Load a trained model.

        Args:
            mobility: Mobility scenario
            channel_model: Channel model type
            modulation: Modulation scheme
            scheme: Training scheme
            snr: SNR value used for training
            model_params: Dictionary of model parameters
            device: Device to load model to

        Returns:
            Loaded DNN model
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        model = DNN(
            input_size=model_params['input_size'],
            hidden_layers=model_params['hidden_layers'],
            output_size=model_params['output_size']
        )

        # Load trained weights
        model_path = PathManager.get_model_path(
            mobility, channel_model, modulation, scheme, snr
        )

        if not model_path.exists():
            raise FileNotFoundError(f"No trained model found at {model_path}")

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        return model

    @staticmethod
    def save_model(
            model: DNN,
            mobility: str,
            channel_model: str,
            modulation: str,
            scheme: str,
            snr: int,
            is_checkpoint: bool = False
    ) -> None:
        """
        Save model weights.

        Args:
            is_checkpoint: If True, saves to checkpoints directory
        """
        if is_checkpoint:
            save_dir = PathManager.MODELS_DIR / "checkpoints"
        else:
            save_dir = PathManager.MODELS_DIR / "trained"

        save_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{mobility}_{channel_model}_{modulation}_{scheme}_DNN_{snr}.pt"
        save_path = save_dir / filename

        torch.save(model.state_dict(), save_path)