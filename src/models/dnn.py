"""
Deep Neural Network model for channel estimation.
This module contains the DNN implementation used in the GRACE framework.
"""

import torch
import torch.nn as nn
from typing import List


class DNN(nn.Module):
    """
    Deep Neural Network for channel estimation.
    """

    def __init__(
            self,
            input_size: int,
            hidden_layers: List[int],
            output_size: int
    ):
        """
        Initialize DNN model.

        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            output_size: Number of output features
        """
        super(DNN, self).__init__()

        # Define layers to match the saved model state dict
        self.layer1 = nn.Linear(input_size, hidden_layers[0])
        self.layer2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.layer3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.layer4 = nn.Linear(hidden_layers[2], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        return x