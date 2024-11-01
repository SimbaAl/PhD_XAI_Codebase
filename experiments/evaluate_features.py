"""
Script for evaluating feature importance and model performance analysis.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from scipy.io import loadmat
import argparse

from src.models import DNN
from src.feature_analysis import FeatureCategorizer, FeatureEvaluator
from src.feature_analysis.categorization import CategoryThresholds
from src.utils import PathManager


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Feature Analysis Script')

    # Dataset parameters
    parser.add_argument('--mobility', type=str, required=True,
                        help='Mobility scenario (e.g., High)')
    parser.add_argument('--channel-model', type=str, required=True,
                        help='Channel model type (e.g., VTV_SDWW)')
    parser.add_argument('--modulation', type=str, required=True,
                        help='Modulation scheme (e.g., 16QAM)')
    parser.add_argument('--scheme', type=str, required=True,
                        help='Training scheme (e.g., TRFI)')
    parser.add_argument('--training-snr', type=str, required=True,
                        help='SNR value used for training')

    # Model parameters
    parser.add_argument('--input-size', type=int, default=104,
                        help='Input size of the model')
    parser.add_argument('--hidden-sizes', type=int, nargs=3, default=[23, 29, 21],
                        help='Sizes of hidden layers')
    parser.add_argument('--output-size', type=int, default=104,
                        help='Output size of the model')

    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> DNN:
    """Load the trained DNN model."""
    model = DNN(
        input_size=args.input_size,
        hidden_layers=args.hidden_sizes,
        output_size=args.output_size
    )

    model_path = PathManager.get_model_path(
        mobility=args.mobility,
        channel_model=args.channel_model,
        modulation=args.modulation,
        scheme=args.scheme,
        snr=args.training_snr
    )

    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def load_relevance_scores(args: argparse.Namespace) -> np.ndarray:
    """Load pre-computed relevance scores."""
    relevance_path = PathManager.DATA_DIR / "processed" / "relevance_scores" / \
                     f"{args.mobility}_{args.channel_model}_{args.modulation}_{args.scheme}_relevance_scores.npy"

    if not relevance_path.exists():
        raise FileNotFoundError(f"No relevance scores found at {relevance_path}")

    return np.load(relevance_path)


def load_test_data(args: argparse.Namespace, snr: int) -> Dict[str, torch.Tensor]:
    """Load test dataset for given SNR."""
    file_path = PathManager.DATA_DIR / "raw" / \
                f"{args.mobility}_{args.channel_model}_{args.modulation}_{args.scheme}_DNN_testing_dataset_{snr}.mat"

    if not file_path.exists():
        raise FileNotFoundError(f"No test data found at {file_path}")

    data = loadmat(str(file_path))
    dataset = data['DNN_Datasets']

    # Extract test data
    X = dataset['Test_X'][0, 0]
    Y = dataset['Test_Y'][0, 0]

    # Convert to torch tensors
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()

    return {'inputs': X, 'targets': Y}


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    results_dir = PathManager.RESULTS_DIR / \
                  f"{args.mobility}_{args.channel_model}_{args.modulation}_{args.scheme}"
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load model and relevance scores
        model = load_model(args, device)
        relevance_scores = load_relevance_scores(args)

        # Initialize analysis modules
        categorizer = FeatureCategorizer(
            thresholds=CategoryThresholds(
                high_positive=0.6,
                low_positive=0.2,
                neutral=0.1
            )
        )
        evaluator = FeatureEvaluator(model, device)

        # Get feature categories
        avg_relevance = np.mean(np.abs(relevance_scores), axis=0)
        feature_indices = categorizer.get_feature_indices(avg_relevance)

        # Plot relevance distribution
        channel_type = 'HFS' if args.channel_model == 'VTV_SDWW' else 'LFS'
        categorizer.plot_relevance_distribution(
            avg_relevance,
            save_path=str(results_dir / 'relevance'),
            channel_type=channel_type
        )

        # Evaluate across SNR values
        snr_values = range(0, 45, 5)
        results = {}

        for snr in snr_values:
            print(f"\nEvaluating SNR: {snr} dB")

            try:
                # Load test data
                data = load_test_data(args, snr)
                inputs = data['inputs'].to(device)
                targets = data['targets'].to(device)

                # Evaluate all categories
                results[snr] = evaluator.evaluate_categories(
                    inputs,
                    targets,
                    feature_indices
                )

                # Print current results
                for category, result in results[snr].items():
                    print(f"{category.upper()}:")
                    print(f"  MSE: {result.mse:.6f}")
                    print(f"  RMSE: {result.rmse:.6f}")

            except Exception as e:
                print(f"Error processing SNR {snr}dB: {e}")
                continue

        # Generate analysis and visualizations
        print("\nGenerating analysis and visualizations...")

        # Analyze stability
        stability_metrics = evaluator.analyze_stability(
            results=results,
            output_dir=results_dir
        )
        print("✓ Stability analysis completed")

        # Plot performance trends
        evaluator.plot_performance_trends(
            results=results,
            output_dir=results_dir
        )
        print("✓ Performance trends plotted")

        # Analyze relative contribution
        relative_performance = evaluator.analyze_relative_contribution(
            results=results,
            output_dir=results_dir
        )
        print("✓ Relative contribution analysis completed")

        # Generate summary report
        evaluator.generate_summary_report(
            results=results,
            stability_metrics=stability_metrics,
            relative_performance=relative_performance,
            output_dir=results_dir
        )
        print("✓ Summary report generated")

        print(f"\nAnalysis complete! Results saved to: {results_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")
        return


if __name__ == "__main__":
    main()