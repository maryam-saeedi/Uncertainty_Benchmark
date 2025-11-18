from abc import ABC, abstractmethod
import torch.nn as nn


class Model(nn.Module, ABC):
    """Base model class for all model implementations."""

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

    @abstractmethod
    def forward(self, x):
        """Forward pass through the model. Must be implemented by child classes."""
        pass

