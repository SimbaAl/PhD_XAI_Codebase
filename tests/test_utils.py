"""
Test script for utility functions.
"""

import torch
import sys
from pathlib import Path
import numpy as np

# Add project root to Python path if needed
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils import PathManager, DataLoader, ModelManager


def test_path_manager():
    """Test PathManager functionality."""
    print("\nTesting PathManager...")

    # Test directory creation
    PathManager.create_required_directories()
    print("✓ Created required directories")

    # Verify data existence
    files_exist = PathManager.verify_data_exists(
        mobility="High",
        channel_model="VTV_SDWW",
        modulation="16QAM",
        scheme="TRFI",
        training_snr=40,
        test_snrs=range(0, 45, 5)
    )

    print("\nFile existence check:")
    print(f"Model exists: {files_exist['model']}")
    print(f"Training data exists: {files_exist['training_data']}")
    print("\nTesting data existence:")
    for snr, exists in files_exist['testing_data'].items():
        print(f"SNR {snr}dB: {exists}")


def test_model_manager():
    """Test ModelManager functionality."""
    print("\nTesting ModelManager...")

    # Model parameters
    model_params = {
        'input_size': 104,
        'hidden_layers': [23, 29, 21],
        'output_size': 104
    }

    try:
        # Load model
        model = ModelManager.load_model(
            mobility="High",
            channel_model="VTV_SDWW",
            modulation="16QAM",
            scheme="TRFI",
            snr=40,
            model_params=model_params
        )
        print("✓ Successfully loaded model")
        print(f"Model device: {next(model.parameters()).device}")

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return


def test_data_loader():
    """Test DataLoader functionality."""
    print("\nTesting DataLoader...")

    try:
        # Load training data
        X_train, Y_train = DataLoader.load_training_data(
            mobility="High",
            channel_model="VTV_SDWW",
            modulation="16QAM",
            scheme="TRFI",
            snr=40
        )
        print("\nTraining data loaded successfully:")
        print(f"X_train shape: {X_train.shape}")
        print(f"Y_train shape: {Y_train.shape}")
        print(f"X_train device: {X_train.device}")
        print(f"Y_train device: {Y_train.device}")

        # Load testing data for SNR=0
        X_test, Y_test = DataLoader.load_testing_data(
            mobility="High",
            channel_model="VTV_SDWW",
            modulation="16QAM",
            scheme="TRFI",
            snr=0
        )
        print("\nTesting data loaded successfully:")
        print(f"X_test shape: {X_test.shape}")
        print(f"Y_test shape: {Y_test.shape}")
        print(f"X_test device: {X_test.device}")
        print(f"Y_test device: {Y_test.device}")

        # Test data statistics
        print("\nData statistics:")
        print(f"X_train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
        print(f"Y_train range: [{Y_train.min():.4f}, {Y_train.max():.4f}]")
        print(f"X_test range: [{X_test.min():.4f}, {X_test.max():.4f}]")
        print(f"Y_test range: [{Y_test.min():.4f}, {Y_test.max():.4f}]")

    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return


def main():
    """Run all tests."""
    print("Starting utility tests...")

    # Test PathManager
    test_path_manager()

    # Test ModelManager
    test_model_manager()

    # Test DataLoader
    test_data_loader()

    print("\nTests completed!")


if __name__ == "__main__":
    main()