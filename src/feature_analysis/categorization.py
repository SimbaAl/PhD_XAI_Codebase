"""
Feature categorization based on relevance scores.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class CategoryThresholds:
    """Thresholds for categorizing features based on relevance scores."""
    high_positive: float = 0.6
    low_positive: float = 0.2
    neutral: float = 0.1


class FeatureCategorizer:
    """
    Categorizes features based on their relevance scores.

    Attributes:
        thresholds (CategoryThresholds): Thresholds for feature categorization
    """

    def __init__(self, thresholds: Optional[CategoryThresholds] = None):
        """
        Initialize the categorizer with given thresholds.

        Args:
            thresholds: Custom thresholds for categorization. If None, uses defaults.
        """
        self.thresholds = thresholds or CategoryThresholds()

    def get_feature_indices(self, relevance_scores: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get indices for each feature category based on relevance scores.

        Args:
            relevance_scores: Array of relevance scores for features

        Returns:
            Dictionary mapping category names to feature indices
        """
        high_positive = np.where(relevance_scores > self.thresholds.high_positive)[0]
        low_positive = np.where(
            (relevance_scores >= self.thresholds.low_positive) &
            (relevance_scores <= self.thresholds.high_positive)
        )[0]
        neutral = np.where(
            (relevance_scores >= self.thresholds.neutral) &
            (relevance_scores < self.thresholds.low_positive)
        )[0]
        negative = np.where(relevance_scores < self.thresholds.neutral)[0]

        # Combined categories
        positive = np.concatenate([high_positive, low_positive])

        return {
            'full': np.arange(len(relevance_scores)),
            'high_positive': high_positive,
            'low_positive': low_positive,
            'positive': positive,
            'neutral': neutral,
            'negative': negative,
            'positive_neutral': np.concatenate([positive, neutral]),
            'negative_neutral': np.concatenate([negative, neutral])
        }

    def plot_relevance_distribution(
            self,
            relevance_scores: np.ndarray,
            save_path: str,
            channel_type: str = "Unknown",
            normalize: bool = True
    ) -> None:
        """
        Create histogram of relevance score distribution.

        Args:
            relevance_scores: Array of relevance scores
            save_path: Path to save the plot
            channel_type: Type of channel for plot title
            normalize: Whether to normalize scores to [0,1] range
        """
        scores = np.abs(relevance_scores)

        if normalize:
            scores = (scores - scores.min()) / (scores.max() - scores.min())
            scores = 1 - scores  # Complement to match paper's representation

        # Create histogram bins
        bins = np.arange(0, 1.1, 0.1)
        hist, _ = np.histogram(scores, bins=bins)

        plt.figure(figsize=(12, 8))
        bar_width = 0.08
        x = bins[:-1]

        # Create bar plot
        plt.bar(x, hist, width=bar_width, alpha=0.7,
                label='TRFI-MLP Method', align='edge')

        # Add value labels
        for i, v in enumerate(hist):
            if v > 0:
                plt.text(x[i] + bar_width / 2, v, str(int(v)),
                         ha='center', va='bottom')

        plt.xlabel('Normalised LRP scores')
        plt.ylabel('Subcarriers')
        plt.title(f'Relevance Score Distribution - {channel_type}')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.savefig(f'{save_path}_relevance_distribution.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

    def save_categories(
            self,
            feature_indices: Dict[str, np.ndarray],
            base_path: str
    ) -> None:
        """
        Save feature indices for each category.

        Args:
            feature_indices: Dictionary of category indices
            base_path: Base path for saving files
        """
        for category, indices in feature_indices.items():
            np.save(f'{base_path}_{category}_features.npy', indices)
