"""
Tests for the DNN model.
"""

import torch
import pytest
from src.models import DNN


def test_dnn_initialization():
    """Test DNN model initialization."""
    model = DNN(
        input_size=104,
        hidden_layers=[23, 29, 21],
        output_size=104
    )

    assert model.input_size == 104
    assert model.output_size == 104
    assert model.hidden_layers == [23, 29, 21]


def test_dnn_forward():
    """Test DNN forward pass."""
    model = DNN(
        input_size=104,
        hidden_layers=[23, 29, 21],
        output_size=104
    )

    # Create dummy input
    batch_size = 32
    x = torch.randn(batch_size, 104)

    # Forward pass
    output = model(x)

    # Check output shape
    assert output.shape == (batch_size, 104)


def test_dnn_config():
    """Test DNN configuration methods."""
    original_model = DNN(
        input_size=104,
        hidden_layers=[23, 29, 21],
        output_size=104
    )

    # Get config
    config = original_model.get_config()

    # Create new model from config
    new_model = DNN.from_config(config)

    # Check configurations match
    assert new_model.input_size == original_model.input_size
    assert new_model.hidden_layers == original_model.hidden_layers
    assert new_model.output_size == original_model.output_size