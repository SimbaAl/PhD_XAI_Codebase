"""
Example script demonstrating the use of feature analysis modules.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
import h5py
import argparse
from src.models import DNN
from src.feature_analysis import FeatureCategorizer, FeatureEvaluator
from src.feature_analysis.categorization import CategoryThresholds

# Define project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Feature Analysis Script')
    parser.add_argument('--mobility', type=str, required=True)
    parser.add_argument('--channel-model', type=str, required=True)
    parser.add_argument('--modulation-order', type=str, required=True)
    parser.add_argument('--scheme', type=str, required=True)
    parser.add_argument('--training-snr', type=str, required=True)
    parser.add_argument('--input-size', type=int, default=104)
    parser.add_argument('--hidden-sizes', type=int, nargs=3, default=[23, 29, 21])
    parser.add_argument('--output-size', type=int, default=104)
    return parser.parse_args()

def load_model(args: argparse.Namespace, device: torch.device) -> DNN:
    """Load the trained DNN model."""
    model = DNN(
        input_size=args.input_size,
        hidden_layers=args.hidden_sizes,
        output_size=args.output_size
    )

    # Construct model path using trained models directory
    model_filename = (f'{args.mobility}_{args.channel_model}_{args.modulation_order}'
                     f'_{args.scheme}_DNN_{args.training_snr}.pt')
    model_path = MODELS_DIR / "trained" / model_filename

    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def load_relevance_scores(args: argparse.Namespace) -> np.ndarray:
    """Load pre-computed relevance scores."""
    # Store relevance scores in processed data directory
    rel_scores_filename = (f'{args.mobility}_{args.channel_model}_{args.modulation_order}'
                         f'_{args.scheme}_relevance_scores.npy')
    file_path = DATA_DIR / "processed" / "relevance_scores" / rel_scores_filename

    if not file_path.exists():
        raise FileNotFoundError(f"No relevance scores found at {file_path}")

    return np.load(file_path)

def load_test_data(args: argparse.Namespace, snr: int) -> Dict[str, torch.Tensor]:
    """Load test dataset for given SNR."""
    # Test data should be in processed/training_data directory
    filename = (f'{args.mobility}_{args.channel_model}_{args.modulation_order}'
               f'_{args.scheme}_DNN_testing_dataset_{snr}.mat')
    file_path = DATA_DIR / "processed" / "training_data" / filename

    if not file_path.exists():
        raise FileNotFoundError(f"No test data found at {file_path}")

    with h5py.File(file_path, 'r') as file:
        dataset = file['DNN_Datasets'][0, 0]
        X = torch.from_numpy(dataset['Test_X'][:]).float()
        Y = torch.from_numpy(dataset['Test_Y'][:]).float()

    return {'inputs': X, 'targets': Y}

def main():
    # Parse arguments and setup
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create necessary directories if they don't exist
    for dir_path in [DATA_DIR / "processed" / "relevance_scores",
                    DATA_DIR / "processed" / "training_data",
                    MODELS_DIR / "trained",
                    RESULTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Create experiment-specific results directory
    results_dir = RESULTS_DIR / f'{args.mobility}_{args.channel_model}_{args.modulation_order}_{args.scheme}'
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load model and relevance scores
        model = load_model(args, device)
        relevance_scores = load_relevance_scores(args)

        # Initialize feature analysis modules
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
            save_path=results_dir / 'relevance',
            channel_type=channel_type
        )

        # Save feature categories
        categorizer.save_categories(
            feature_indices,
            base_path=results_dir / 'features'
        )

        # Evaluate across SNR values
        snr_values = range(0, 45, 5)
        results = {}

        for snr in snr_values:
            print(f"\nEvaluating SNR: {snr} dB")

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

        # Generate analysis and visualizations
        stability_metrics = evaluator.analyze_stability(results)

        evaluator.plot_performance_trends(
            results,
            save_path=results_dir / 'performance'
        )

        relative_performance = evaluator.analyze_relative_contribution(
            results,
            save_path=results_dir / 'contribution'
        )

        evaluator.generate_summary_report(
            results,
            stability_metrics,
            relative_performance,
            save_path=results_dir / 'analysis'
        )

        print("\nAnalysis complete! Results saved to:", results_dir)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all required data files and models are in their respective directories.")
        return

    except Exception as e:
        print(f"An error occurred: {e}")
        return

if __name__ == "__main__":
    main()