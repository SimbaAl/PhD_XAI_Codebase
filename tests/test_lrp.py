"""
Tests for LRP implementation.
"""

import torch
import pytest
from src.models import DNN
from src.lrp import LRP


@pytest.fixture
def model():
    """Create a simple DNN model for testing."""
    return DNN(
        input_size=104,
        hidden_layers=[23, 29, 21],
        output_size=104
    )


@pytest.fixture
def lrp_analyzer(model):
    """Create LRP analyzer instance."""
    return LRP(model)


def test_lrp_initialization(model, lrp_analyzer):
    """Test LRP analyzer initialization."""
    assert lrp_analyzer.model == model
    assert lrp_analyzer.epsilon == 1e-9


def test_relevance_computation(lrp_analyzer):
    """Test relevance score computation."""
    # Create dummy input
    batch_size = 32
    inputs = torch.randn(batch_size, 104)

    # Compute relevance scores
    relevance, activations = lrp_analyzer.compute_relevance(inputs)

    # Check shapes
    assert relevance.shape == inputs.shape
    assert len(activations) == len(lrp_analyzer.model.layers) + 1


def test_conservation_property(lrp_analyzer):
    """Test conservation property of LRP."""
    batch_size = 32
    inputs = torch.randn(batch_size, 104)

    # Get model output and relevance scores
    with torch.no_grad():
        outputs = lrp_analyzer.model(inputs)
    relevance, activations = lrp_analyzer.compute_relevance(inputs)

    # Check if sum of relevance scores is approximately preserved
    # Compare with model outputs instead of inputs
    output_sum = torch.sum(outputs, dim=1)
    relevance_sum = torch.sum(relevance, dim=1)

    assert torch.allclose(output_sum, relevance_sum, rtol=1e-5, atol=1e-5)


def test_feature_importance_analysis(lrp_analyzer):
    """Test feature importance analysis."""
    batch_size = 32
    inputs = torch.randn(batch_size, 104)

    importance_scores = lrp_analyzer.analyze_feature_importance(inputs)

    assert importance_scores.shape == (104,)
    assert torch.all(importance_scores >= 0)  # Scores should be non-negative