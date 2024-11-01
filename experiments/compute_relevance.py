"""
Script for computing Layer-wise Relevance Propagation (LRP) scores.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

from src.models import DNN
from src.utils import ModelManager, DataLoader, PathManager
from src.lrp import LRP


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compute LRP relevance scores')

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
                        help='SNR value used for training')

    # Model parameters
    parser.add_argument('--input-size', type=int, default=104,
                        help='Input size of the model')
    parser.add_argument('--hidden-sizes', type=int, nargs=3,
                        default=[23, 29, 21],
                        help='Sizes of hidden layers')
    parser.add_argument('--output-size', type=int, default=104,
                        help='Output size of the model')

    # LRP parameters
    parser.add_argument('--epsilon', type=float, default=1e-9,
                        help='Epsilon value for LRP computation')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for processing')

    return parser.parse_args()


def compute_batch_relevance(
        lrp_analyzer: LRP,
        inputs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute relevance scores for a batch of inputs.

    Args:
        lrp_analyzer: LRP analyzer instance
        inputs: Input tensor

    Returns:
        Tuple of (relevance scores, predictions)
    """
    relevance, activations = lrp_analyzer.compute_relevance(inputs)
    predictions = activations[-1]
    return relevance, predictions


def process_dataset(
        lrp_analyzer: LRP,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
) -> Dict[str, np.ndarray]:
    """
    Process entire dataset and compute relevance scores.

    Args:
        lrp_analyzer: LRP analyzer instance
        dataloader: DataLoader instance
        device: Device to use for computation

    Returns:
        Dictionary containing relevance scores and related data
    """
    all_relevance = []
    all_predictions = []
    all_inputs = []

    print("\nComputing relevance scores...")
    for batch_idx, (inputs, _) in enumerate(dataloader):
        inputs = inputs.to(device)

        # Compute relevance scores
        relevance, predictions = compute_batch_relevance(lrp_analyzer, inputs)

        # Store results
        all_relevance.append(relevance.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_inputs.append(inputs.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1} batches...")

    # Concatenate results
    results = {
        'relevance_scores': np.concatenate(all_relevance, axis=0),
        'model_predictions': np.concatenate(all_predictions, axis=0),
        'original_inputs': np.concatenate(all_inputs, axis=0)
    }

    return results


def plot_relevance_distribution(
        relevance_scores: np.ndarray,
        save_path: Path,
        channel_type: str
) -> None:
    """
    Create visualization of relevance score distribution.

    Args:
        relevance_scores: Array of relevance scores
        save_path: Path to save the plot
        channel_type: Type of channel for plot title
    """
    # Calculate average absolute relevance scores
    avg_relevance = np.mean(np.abs(relevance_scores), axis=0)

    # Normalize scores
    scores = (avg_relevance - avg_relevance.min()) / (avg_relevance.max() - avg_relevance.min())
    scores = 1 - scores  # Complement to match paper's representation

    # Create histogram
    plt.figure(figsize=(12, 8))
    plt.hist(scores, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Normalized Relevance Score')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Relevance Scores - {channel_type}')
    plt.grid(True, alpha=0.3)

    # Save plot
    plt.savefig(save_path / 'relevance_distribution.png',
                bbox_inches='tight', dpi=300)
    plt.close()


def main():
    """Main function for computing relevance scores."""
    args = parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    PathManager.create_required_directories()

    def get_save_directory(args: argparse.Namespace) -> Path:
        """Get directory for saving processed data."""
        processed_dir = PathManager.DATA_DIR / "processed"
        save_path = processed_dir / f"{args.mobility}_{args.channel_model}_{args.modulation}_{args.scheme}"
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path

    try:
        # Load model
        print("\nLoading model...")
        model_params = {
            'input_size': args.input_size,
            'hidden_layers': args.hidden_sizes,
            'output_size': args.output_size
        }

        model = ModelManager.load_model(
            mobility=args.mobility,
            channel_model=args.channel_model,
            modulation=args.modulation,
            scheme=args.scheme,
            snr=args.training_snr,
            model_params=model_params,
            device=device
        )
        print("✓ Model loaded successfully")

        # Initialize LRP analyzer
        lrp_analyzer = LRP(model, epsilon=args.epsilon)
        print("✓ LRP analyzer initialized")

        # Load validation data for computing relevance
        print("\nLoading validation data...")
        X, Y = DataLoader.load_training_data(
            mobility=args.mobility,
            channel_model=args.channel_model,
            modulation=args.modulation,
            scheme=args.scheme,
            snr=args.training_snr
        )

        # Create dataloader
        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        print("✓ Data loaded successfully")

        # Compute relevance scores
        results = process_dataset(lrp_analyzer, dataloader, device)

        # Save results
        print("\nSaving results...")
        save_path = get_save_directory(args)

        np.save(save_path / f"relevance_scores_{args.training_snr}.npy",
                results['relevance_scores'])
        np.save(save_path / f"model_predictions_{args.training_snr}.npy",
                results['model_predictions'])
        np.save(save_path / f"original_inputs_{args.training_snr}.npy",
                results['original_inputs'])

        # Create visualization
        channel_type = 'HFS' if args.channel_model == 'VTV_SDWW' else 'LFS'
        plot_relevance_distribution(
            results['relevance_scores'],
            save_path,
            channel_type
        )

        print("\nRelevance computation completed successfully!")
        print(f"Results saved to: {save_path}")

    except Exception as e:
        print(f"\nError occurred: {e}")
        raise


if __name__ == "__main__":
    main()