from typing import Dict, Type
import torch.nn as nn
from omegaconf import DictConfig

from methods.method import Method


class MethodFactory:
    """Factory for creating method instances based on config."""

    _registry: Dict[str, Type[Method]] = {}

    @classmethod
    def register(cls, name: str, method_class: Type[Method]):
        """Register a method class with a name."""
        cls._registry[name] = method_class

    @classmethod
    def create(cls, config: DictConfig, **kwargs) -> Method:
        """Create a method instance based on config.

        Args:
            config: Hydra configuration containing method specifications
            **kwargs: Additional keyword arguments to pass to method constructor

        Returns:
            Instantiated method
        """
        method_name = config.method.name.lower()

        if method_name not in cls._registry:
            raise ValueError(
                f"Unknown method: {method_name}. "
                f"Available methods: {list(cls._registry.keys())}"
            )

        return cls._registry[method_name](config, **kwargs)

    @classmethod
    def get_available_methods(cls) -> list:
        """Get list of available method names."""
        return list(cls._registry.keys())


# Decorator for easy registration
def register_method(name: str):
    """Decorator to register a method class."""
    def decorator(method_class: Type[Method]):
        MethodFactory.register(name, method_class)
        return method_class
    return decorator
