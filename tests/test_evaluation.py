"""
Test script for feature evaluation module.
"""

import torch
import sys
from pathlib import Path
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.feature_analysis import FeatureEvaluator, FeatureCategorizer
from src.utils import ModelManager, DataLoader, PathManager
from src.feature_analysis.categorization import CategoryThresholds

def test_feature_evaluation():
    """Test the complete feature evaluation pipeline."""
    print("\nTesting Feature Evaluation Pipeline...")

    # Setup parameters
    mobility = "High"
    channel_model = "VTV_SDWW"
    modulation = "16QAM"
    scheme = "TRFI"
    training_snr = 40

    # Model parameters
    model_params = {
        'input_size': 104,
        'hidden_layers': [23, 29, 21],
        'output_size': 104
    }

    try:
        # 1. Load Model
        print("\n1. Loading model...")
        model = ModelManager.load_model(
            mobility=mobility,
            channel_model=channel_model,
            modulation=modulation,
            scheme=scheme,
            snr=training_snr,
            model_params=model_params
        )
        print("✓ Model loaded successfully")

        # 2. Initialize Evaluator
        print("\n2. Initializing feature evaluator...")
        evaluator = FeatureEvaluator(model)
        print("✓ Evaluator initialized")

        # 3. Load and process relevance scores
        print("\n3. Loading relevance scores...")
        # Check if relevance scores exist, if not, create dummy scores for testing
        try:
            relevance_scores = DataLoader.load_processed_data(
                mobility=mobility,
                channel_model=channel_model,
                modulation=modulation,
                scheme=scheme,
                data_type="relevance_scores",
                snr=training_snr
            )
        except FileNotFoundError:
            print("   No relevance scores found, creating dummy scores for testing...")
            relevance_scores = np.random.rand(1000, 104)

        avg_relevance = np.mean(np.abs(relevance_scores), axis=0)
        print("✓ Relevance scores processed")

        # 4. Get feature categories
        print("\n4. Categorizing features...")
        categorizer = FeatureCategorizer(
            thresholds=CategoryThresholds(
                high_positive=0.6,
                low_positive=0.2,
                neutral=0.1
            )
        )
        feature_indices = categorizer.get_feature_indices(avg_relevance)
        print("✓ Features categorized")
        print(f"   Number of categories: {len(feature_indices)}")
        for cat, indices in feature_indices.items():
            print(f"   {cat}: {len(indices)} features")

        # 5. Evaluate across SNR values
        print("\n5. Evaluating performance across SNR values...")
        results = {}
        snr_values = range(0, 45, 5)

        for snr in snr_values:
            print(f"\nProcessing SNR {snr}dB...")
            try:
                # Load test data for this SNR
                X_test, Y_test = DataLoader.load_testing_data(
                    mobility=mobility,
                    channel_model=channel_model,
                    modulation=modulation,
                    scheme=scheme,
                    snr=snr
                )

                # Evaluate categories
                results[snr] = evaluator.evaluate_categories(
                    X_test,
                    Y_test,
                    feature_indices
                )

                print(f"   Results for SNR {snr}dB:")
                for category, result in results[snr].items():
                    print(f"   {category.upper()}:")
                    print(f"     MSE: {result.mse:.6f}")
                    print(f"     RMSE: {result.rmse:.6f}")

            except Exception as e:
                print(f"Error processing SNR {snr}dB: {e}")
                continue

        # 6. Generate analysis and visualizations
        print("\n6. Generating analysis and visualizations...")
        output_dir = project_root / "results" / f"{mobility}_{channel_model}_{modulation}_{scheme}"

        try:
            # Analyze stability
            stability_metrics = evaluator.analyze_stability(results, output_dir)
            print("✓ Stability analysis completed")

            # Plot performance trends
            evaluator.plot_performance_trends(results, output_dir)
            print("✓ Performance trends plotted")

            # Analyze relative contribution
            relative_performance = evaluator.analyze_relative_contribution(
                results, output_dir
            )
            print("✓ Relative contribution analysis completed")

            # Generate summary report
            evaluator.generate_summary_report(
                results,
                stability_metrics,
                relative_performance,
                output_dir
            )
            print("✓ Summary report generated")

            print(f"\nResults saved to: {output_dir}")

        except Exception as e:
            print(f"Error in analysis generation: {e}")
            raise

    except Exception as e:
        print(f"\nTest failed: {e}")
        raise

    print("\nFeature evaluation test completed successfully!")

if __name__ == "__main__":
    test_feature_evaluation()