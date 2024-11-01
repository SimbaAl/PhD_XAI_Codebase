"""
Evaluation module for analyzing model performance across different feature categories.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from pathlib import Path
import json


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    mse: float
    rmse: float
    predictions: np.ndarray


@dataclass
class StabilityMetrics:
    """Container for stability analysis metrics."""
    mean_mse: float
    std_mse: float
    cv: float  # Coefficient of variation


class FeatureEvaluator:
    """
    Evaluates model performance across different feature categories and SNR values.
    """

    def __init__(
            self,
            model: nn.Module,
            device: Optional[torch.device] = None
    ):
        """Initialize the evaluator."""
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()

    def _setup_directories(self, base_path: Path) -> Dict[str, Path]:
        """Create directory structure for outputs."""
        dirs = {
            'base': base_path,
            'visualizations': base_path / 'visualizations',
            'metrics': base_path / 'metrics',
            'reports': base_path / 'reports',
            'plots': {
                'performance': base_path / 'visualizations' / 'performance',
                'contribution': base_path / 'visualizations' / 'contribution'
            },
            'metrics': {
                'stability': base_path / 'metrics' / 'stability',
                'performance': base_path / 'metrics' / 'performance',
                'relative': base_path / 'metrics' / 'relative'
            }
        }

        # Create all directories
        for path in dirs.values():
            if isinstance(path, dict):
                for subpath in path.values():
                    subpath.mkdir(parents=True, exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)

        return dirs

    def evaluate_model(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:
        """Evaluate model performance on given inputs and targets."""
        with torch.no_grad():
            outputs = self.model(inputs)
            mse_loss = nn.MSELoss()(outputs, targets).item()
            rmse = np.sqrt(mean_squared_error(
                targets.cpu().numpy(),
                outputs.cpu().numpy()
            ))
        return outputs, mse_loss, rmse

    def evaluate_categories(
            self,
            X_test: torch.Tensor,
            Y_test: torch.Tensor,
            feature_indices: Dict[str, np.ndarray]
    ) -> Dict[str, EvaluationResult]:
        """Evaluate model performance for each feature category."""
        results = {}

        for category, indices in feature_indices.items():
            if category == 'full':
                masked_input = X_test
            else:
                masked_input = torch.zeros_like(X_test)
                masked_input[:, indices] = X_test[:, indices]

            outputs, mse, rmse = self.evaluate_model(masked_input, Y_test)

            results[category] = EvaluationResult(
                mse=float(mse),
                rmse=float(rmse),
                predictions=outputs.cpu().numpy()
            )

        return results

    def analyze_stability(
            self,
            results: Dict[int, Dict[str, EvaluationResult]],
            output_dir: Path,
            snr_regions: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> Dict[str, Dict[str, StabilityMetrics]]:
        """Analyze performance stability across SNR regions."""
        dirs = self._setup_directories(output_dir)

        if snr_regions is None:
            snr_regions = {
                'low': (0, 15),
                'medium': (20, 30),
                'high': (35, 40)
            }

        stability_metrics = {}
        categories = list(next(iter(results.values())).keys())

        for cat in categories:
            stability_metrics[cat] = {}
            for region, (start, end) in snr_regions.items():
                region_snrs = [
                    snr for snr in results.keys()
                    if start <= snr <= end
                ]
                mse_values = [
                    float(results[snr][cat].mse)
                    for snr in region_snrs
                ]

                mean_mse = float(np.mean(mse_values))
                std_mse = float(np.std(mse_values))
                cv = float((std_mse / mean_mse * 100) if mean_mse != 0 else float('inf'))

                metrics = StabilityMetrics(mean_mse=mean_mse, std_mse=std_mse, cv=cv)
                stability_metrics[cat][region] = metrics

                # Save individual metric files
                metric_file = dirs['metrics']['stability'] / f"{cat}_{region}_metrics.json"
                with open(metric_file, 'w') as f:
                    json.dump({
                        'mean_mse': mean_mse,
                        'std_mse': std_mse,
                        'cv': cv
                    }, f, indent=4)

        return stability_metrics

    def plot_performance_trends(
            self,
            results: Dict[int, Dict[str, EvaluationResult]],
            output_dir: Path
    ) -> None:
        """Create visualizations of performance trends across SNRs."""
        dirs = self._setup_directories(output_dir)
        snr_values = sorted(results.keys())
        categories = list(results[snr_values[0]].keys())

        # Prepare data
        mse_data = {
            cat: [float(results[snr][cat].mse) for snr in snr_values]
            for cat in categories
        }
        rmse_data = {
            cat: [float(results[snr][cat].rmse) for snr in snr_values]
            for cat in categories
        }

        # Save performance metrics
        performance_file = dirs['metrics']['performance'] / 'snr_performance.json'
        with open(performance_file, 'w') as f:
            json.dump({
                'mse': mse_data,
                'rmse': rmse_data,
                'snr_values': list(snr_values)
            }, f, indent=4)

        # Plot MSE trends
        plt.figure(figsize=(12, 8))
        for cat in categories:
            plt.semilogy(snr_values, mse_data[cat], marker='o', label=cat.upper())
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel('SNR (dB)')
        plt.ylabel('MSE (log scale)')
        plt.title('MSE Performance Across SNR Values')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(dirs['plots']['performance'] / 'mse_trends.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

        # Plot RMSE trends
        plt.figure(figsize=(12, 8))
        for cat in categories:
            plt.semilogy(snr_values, rmse_data[cat], marker='o', label=cat.upper())
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel('SNR (dB)')
        plt.ylabel('RMSE (log scale)')
        plt.title('RMSE Performance Across SNR Values')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(dirs['plots']['performance'] / 'rmse_trends.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

    def analyze_relative_contribution(
            self,
            results: Dict[int, Dict[str, EvaluationResult]],
            output_dir: Path
    ) -> Dict[int, Dict[str, float]]:
        """Calculate and visualize the relative contribution of each category."""
        dirs = self._setup_directories(output_dir)
        snr_values = sorted(results.keys())
        categories = list(results[snr_values[0]].keys())

        # Calculate relative performance
        relative_performance = {snr: {} for snr in snr_values}
        for snr in snr_values:
            baseline_mse = float(results[snr]['positive_neutral'].mse)
            for cat in categories:
                relative_performance[snr][cat] = float(
                    (baseline_mse / float(results[snr][cat].mse)) * 100
                )

        # Save relative performance metrics
        relative_file = dirs['metrics']['relative'] / 'relative_performance.json'
        with open(relative_file, 'w') as f:
            json.dump(relative_performance, f, indent=4)

        # Create heatmap
        performance_matrix = np.zeros((len(categories), len(snr_values)))
        for i, cat in enumerate(categories):
            for j, snr in enumerate(snr_values):
                performance_matrix[i, j] = relative_performance[snr][cat]

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            performance_matrix,
            xticklabels=snr_values,
            yticklabels=[cat.upper() for cat in categories],
            annot=True,
            fmt='.1f',
            cmap='YlOrRd'
        )
        plt.xlabel('SNR (dB)')
        plt.ylabel('Category')
        plt.title('Relative Performance Contribution (%)')
        plt.tight_layout()
        plt.savefig(dirs['plots']['contribution'] / 'relative_contribution.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

        return relative_performance

    def generate_summary_report(
            self,
            results: Dict[int, Dict[str, EvaluationResult]],
            stability_metrics: Dict[str, Dict[str, StabilityMetrics]],
            relative_performance: Dict[int, Dict[str, float]],
            output_dir: Path
    ) -> None:
        """Generate a comprehensive summary report."""
        dirs = self._setup_directories(output_dir)
        report_path = dirs['reports'] / 'summary_report.txt'

        with open(report_path, 'w') as f:
            f.write("Performance Analysis Summary Report\n")
            f.write("================================\n\n")

            # Overall Performance Rankings
            snr_values = sorted(results.keys())
            categories = list(results[snr_values[0]].keys())

            avg_performance = {
                cat: float(np.mean([float(results[snr][cat].mse) for snr in snr_values]))
                for cat in categories
            }

            ranked_categories = sorted(avg_performance.items(), key=lambda x: x[1])

            f.write("1. Overall Performance Rankings\n")
            f.write("--------------------------\n")
            for rank, (cat, mse) in enumerate(ranked_categories, 1):
                f.write(f"{rank}. {cat.upper()}: Average MSE = {mse:.6f}\n")

            # Performance Improvement Analysis
            f.write("\n2. Performance Improvement Analysis\n")
            f.write("--------------------------------\n")

            improvement_metrics = {}
            for cat in categories:
                initial_mse = float(results[min(snr_values)][cat].mse)
                final_mse = float(results[max(snr_values)][cat].mse)
                improvement = float(((initial_mse - final_mse) / initial_mse) * 100)

                improvement_metrics[cat] = {
                    'initial_mse': initial_mse,
                    'final_mse': final_mse,
                    'improvement_percentage': improvement
                }

                f.write(f"\n{cat.upper()}:")
                f.write(f"\n  Initial MSE ({min(snr_values)} dB): {initial_mse:.6f}")
                f.write(f"\n  Final MSE ({max(snr_values)} dB): {final_mse:.6f}")
                f.write(f"\n  Improvement: {improvement:.1f}%\n")

            # Save improvement metrics
            improvement_file = dirs['metrics']['performance'] / 'improvement_metrics.json'
            with open(improvement_file, 'w') as imf:
                json.dump(improvement_metrics, imf, indent=4)

            # Stability Analysis Summary
            f.write("\n3. Stability Analysis\n")
            f.write("-------------------\n")

            stability_summary = {}
            for cat in categories:
                f.write(f"\n{cat.upper()}:")
                stability_summary[cat] = {}

                for region in ['low', 'medium', 'high']:
                    metrics = stability_metrics[cat][region]
                    cv = float(metrics.cv)
                    f.write(f"\n  {region.upper()} SNR Region:")
                    f.write(f"\n    Mean MSE: {float(metrics.mean_mse):.6f}")
                    f.write(f"\n    Std Dev MSE: {float(metrics.std_mse):.6f}")
                    f.write(f"\n    CV: {cv:.2f}%")

                    stability_summary[cat][region] = {
                        'mean_mse': float(metrics.mean_mse),
                        'std_mse': float(metrics.std_mse),
                        'cv': float(cv)
                    }
                f.write("\n")

            # Save stability summary
            stability_summary_file = dirs['metrics']['stability'] / 'stability_summary.json'
            with open(stability_summary_file, 'w') as ssf:
                json.dump(stability_summary, ssf, indent=4)

            f.write("\nAnalysis files have been saved to the following directories:")
            f.write(f"\n- Visualizations: {dirs['visualizations']}")
            f.write(f"\n- Metrics: {dirs['metrics']}")
            f.write(f"\n- Reports: {dirs['reports']}\n")