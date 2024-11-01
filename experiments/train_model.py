"""
Script for training the DNN model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import time
from typing import Dict, Any

from src.models import DNN
from src.utils import ModelManager, DataLoader, PathManager


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DNN model for channel estimation')

    # Dataset parameters
    parser.add_argument('--mobility', type=str, required=True,
                        help='Mobility scenario (e.g., High)')
    parser.add_argument('--channel-model', type=str, required=True,
                        help='Channel model type (e.g., VTV_SDWW)')
    parser.add_argument('--modulation', type=str, required=True,
                        help='Modulation scheme (e.g., 16QAM)')
    parser.add_argument('--scheme', type=str, required=True,
                        help='Training scheme (e.g., TRFI)')
    parser.add_argument('--training-snr', type=int, required=True,
                        help='SNR value for training')

    # Model parameters
    parser.add_argument('--input-size', type=int, default=104,
                        help='Input size of the model')
    parser.add_argument('--hidden-sizes', type=int, nargs=3,
                        default=[23, 29, 21],
                        help='Sizes of hidden layers')
    parser.add_argument('--output-size', type=int, default=104,
                        help='Output size of the model')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--validation-split', type=float, default=0.2,
                        help='Validation split ratio')

    return parser.parse_args()


def create_dataloaders(
        X: torch.Tensor,
        Y: torch.Tensor,
        batch_size: int,
        val_split: float
) -> Dict[str, torch.utils.data.DataLoader]:
    """Create training and validation dataloaders."""
    # Calculate split index
    split_idx = int(len(X) * (1 - val_split))

    # Split data
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]

    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return {'train': train_loader, 'val': val_loader}


def train_epoch(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def main():
    """Main training function."""
    args = parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    PathManager.create_required_directories()

    # Load training data
    print("\nLoading training data...")
    X, Y = DataLoader.load_training_data(
        mobility=args.mobility,
        channel_model=args.channel_model,
        modulation=args.modulation,
        scheme=args.scheme,
        snr=args.training_snr
    )
    print(f"Training data loaded: {X.shape}, {Y.shape}")

    # Create dataloaders
    dataloaders = create_dataloaders(
        X, Y,
        batch_size=args.batch_size,
        val_split=args.validation_split
    )
    print("Dataloaders created")

    # Initialize model
    model = DNN(
        input_size=args.input_size,
        hidden_layers=args.hidden_sizes,
        output_size=args.output_size
    ).to(device)

    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    start_time = time.time()

    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(
            model, dataloaders['train'],
            criterion, optimizer, device
        )
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(
            model, dataloaders['val'],
            criterion, device
        )
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ModelManager.save_model(
                model=model,
                mobility=args.mobility,
                channel_model=args.channel_model,
                modulation=args.modulation,
                scheme=args.scheme,
                snr=args.training_snr,
                is_checkpoint=False
            )

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            ModelManager.save_model(
                model=model,
                mobility=args.mobility,
                channel_model=args.channel_model,
                modulation=args.modulation,
                scheme=args.scheme,
                snr=args.training_snr,
                is_checkpoint=True
            )

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{args.epochs}]")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Best Val Loss: {best_val_loss:.6f}")
            print(f"Time: {time.time() - start_time:.2f}s")
            print("-" * 50)

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Total training time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()