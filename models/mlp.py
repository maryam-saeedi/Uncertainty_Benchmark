import torch
import torch.nn as nn
from models.model import Model
from models.model_factory import register_model


@register_model("mlp")
class MLP(Model):
    """Multi-Layer Perceptron model with configurable architecture."""

    def __init__(self, config):
        super(MLP, self).__init__(config)

        # Extract configuration
        input_dim = config.dataset.get('input_dim', 784)  # Default for MNIST
        output_dim = config.dataset.get('num_classes', 10)  # Default for MNIST
        hidden_dim = config.model.get('hidden_dim', 512)
        num_layers = config.model.get('num_layers', 3)
        dropout = config.model.get('dropout', 0.0)
        activation = config.model.get('activation', 'ReLU')
        normalization = config.model.get('normalization', None)

        # Build layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if normalization == 'BatchNorm':
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif normalization == 'LayerNorm':
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(self._get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if normalization == 'BatchNorm':
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif normalization == 'LayerNorm':
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        self.network = nn.Sequential(*layers)
        self.fc_output = nn.Linear(hidden_dim, output_dim)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'ELU': nn.ELU(),
            'GELU': nn.GELU(),
            'Tanh': nn.Tanh(),
            'Sigmoid': nn.Sigmoid(),
        }
        return activations.get(activation, nn.ReLU())

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Flatten input if needed (e.g., for image inputs)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        out = self.network(x)
        out = self.fc_output(out)
        return out
