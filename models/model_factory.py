from typing import Dict, Type
from omegaconf import DictConfig

from models.model import Model


class ModelFactory:
    """Factory for creating model instances based on config."""

    _registry: Dict[str, Type[Model]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[Model]):
        """Register a model class with a name."""
        cls._registry[name] = model_class

    @classmethod
    def create(cls, config: DictConfig, **kwargs) -> Model:
        """Create a model instance based on config."""
        model_name = config.model.name.lower()

        if model_name not in cls._registry:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(cls._registry.keys())}"
            )

        return cls._registry[model_name](config, **kwargs)

    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model names."""
        return list(cls._registry.keys())


# Decorator for easy registration
def register_model(name: str):
    """Decorator to register a model class."""
    def decorator(model_class: Type[Model]):
        ModelFactory.register(name, model_class)
        return model_class
    return decorator
