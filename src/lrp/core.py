"""
Layer-wise Relevance Propagation (LRP) implementation.
This module contains the core LRP functionality for analyzing DNN models.
"""

import torch
import torch.nn as nn
from typing import List, Union, Tuple
from src.models.dnn import DNN


class LRP:
    """
    Layer-wise Relevance Propagation implementation for DNN analysis.

    Attributes:
        model (DNN): The neural network model to analyze
        epsilon (float): Small constant for numerical stability
    """

    def __init__(self, model: DNN, epsilon: float = 1e-9):
        """
        Initialize LRP analyzer.

        Args:
            model: Neural network model to analyze
            epsilon: Small constant for numerical stability
        """
        self.model = model
        self.epsilon = epsilon

    def compute_layer_lrp(
            self,
            layer: nn.Linear,
            prev_activations: torch.Tensor,
            next_layer_relevance: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute LRP relevance scores for a single layer.

        Args:
            layer: Current layer
            prev_activations: Activations from previous layer
            next_layer_relevance: Relevance scores from next layer

        Returns:
            Relevance scores for current layer
        """
        with torch.no_grad():
            weights = layer.weight
            bias = layer.bias if layer.bias is not None else torch.zeros(
                layer.out_features,
                device=prev_activations.device
            )

            # Step 1: Forward pass to calculate denominator
            forward_pass = torch.matmul(prev_activations, weights.t()) + bias

            # Improved stabilizer term
            stabilizer = self.epsilon * (
                    torch.sign(forward_pass) *
                    (forward_pass.abs().max() * torch.ones_like(forward_pass))
            )
            denominator = forward_pass + stabilizer

            # Step 2: Element-wise division with safe division
            quotient = torch.where(
                denominator != 0,
                next_layer_relevance / denominator,
                torch.zeros_like(next_layer_relevance)
            )

            # Step 3: Backward pass for numerator
            contributions = torch.matmul(quotient, weights)

            # Step 4: Element-wise product
            relevance = prev_activations * contributions

            # Ensure conservation property with safe division
            sum_next = torch.sum(next_layer_relevance, dim=1, keepdim=True)
            sum_curr = torch.sum(relevance, dim=1, keepdim=True)
            scale_factor = torch.where(
                sum_curr != 0,
                sum_next / (sum_curr + self.epsilon),
                torch.ones_like(sum_curr)
            )
            relevance = relevance * scale_factor

            return relevance

    def compute_relevance(
            self,
            inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Compute relevance scores for input features.

        Args:
            inputs: Input tensor to analyze

        Returns:
            Tuple of (final relevance scores, intermediate activations)
        """
        self.model.eval()

        # Store activations at each layer
        activations = []
        x = inputs

        # Forward pass to collect activations
        with torch.no_grad():
            activations.append(x)

            # Layer 1
            x = self.model.layer1(x)
            x = torch.relu(x)
            activations.append(x)

            # Layer 2
            x = self.model.layer2(x)
            x = torch.relu(x)
            activations.append(x)

            # Layer 3
            x = self.model.layer3(x)
            x = torch.relu(x)
            activations.append(x)

            # Layer 4 (output layer)
            x = self.model.layer4(x)
            activations.append(x)

        # Initialize relevance with the network output
        relevance = activations[-1]

        # Backward propagation of relevance scores
        # Layer 4
        relevance = self.compute_layer_lrp(
            self.model.layer4,
            activations[3],
            relevance
        )

        # Layer 3
        relevance = self.compute_layer_lrp(
            self.model.layer3,
            activations[2],
            relevance
        )

        # Layer 2
        relevance = self.compute_layer_lrp(
            self.model.layer2,
            activations[1],
            relevance
        )

        # Layer 1
        relevance = self.compute_layer_lrp(
            self.model.layer1,
            activations[0],
            relevance
        )

        return relevance, activations

    def analyze_feature_importance(
            self,
            inputs: torch.Tensor,
            aggregate_fn: callable = torch.mean
    ) -> torch.Tensor:
        """
        Analyze feature importance across multiple inputs.

        Args:
            inputs: Batch of input tensors to analyze
            aggregate_fn: Function to aggregate relevance scores across batch

        Returns:
            Aggregated relevance scores for each feature
        """
        relevance_scores, _ = self.compute_relevance(inputs)
        return aggregate_fn(torch.abs(relevance_scores), dim=0)