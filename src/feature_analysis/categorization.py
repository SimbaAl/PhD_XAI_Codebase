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

        # Create histogram bins from 0 to 1
        bins = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Fixed bins
        hist, _ = np.histogram(scores, bins=bins)

        plt.figure(figsize=(12, 8))

        # Create bar plot
        bars = plt.bar(bins[:-1], hist, width=0.2, align='edge',
                       alpha=0.7, color='skyblue', label='Proposed method')

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show label if there are features in the bin
                plt.text(bar.get_x() + bar.get_width() / 2, height,
                         f'{int(height)}',
                         ha='center', va='bottom')

        plt.xlabel('Relevance Score')
        plt.ylabel('Number of subcarriers')
        plt.title(f'Relevance Score Distribution - {channel_type}')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Set x-axis ticks
        plt.xticks(bins)

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
