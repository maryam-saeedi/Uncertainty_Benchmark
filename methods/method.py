from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import os
from models.model_factory import ModelFactory


class Method(ABC):
    """Base method class for all uncertainty quantification methods."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps = 1e-10

    def init_model(self):
        """Initialize the model. Must be implemented by child classes."""
        self.model = ModelFactory.create(self.config)
        self.model.to(self.device)

    def init_optimizer(self):
        """Initialize the optimizer. Must be implemented by child classes."""
        if self.config.optimizer.name == "SGD":
            optimizer_class = torch.optim.SGD
        elif self.config.optimizer.name == "Adam":
            optimizer_class = torch.optim.Adam
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer.name}")
        arguments = OmegaConf.to_container(self.config.optimizer)
        arguments.pop("name")
        arguments.pop("epochs")
        self.optimizer = optimizer_class(
            self.model.parameters(),
            **arguments,
        )

    def build_base_model(self, retrain=False, **kwargs):
        self.init_model()
        self.init_optimizer()
        if retrain:
            assert kwargs['loader'] is not None, "Loader must exists in order to re-train the model."
            self.train_base_model(kwargs['loader'])
            output_save_dir = os.path.join(self.config.method.output.path, 'model')
            os.makedirs(output_save_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(output_save_dir, 'model.pt'))
        else:
            assert kwargs['pretrained'] is not None, "Pretrained checkpoint cannot be None to use a pretrained model."
            self.model.load_state_dict(torch.load(kwargs['pretrained'], weights_only=True))

    def train_base_model(self, loader: torch.utils.data.DataLoader):
        """Train the model using standard supervised learning.
        Args:
            loader: Training data loader
        """
        # Setup optimizer
        optimizer = self.optimizer
        criterion = nn.CrossEntropyLoss()

        # Training
        epochs = self.config.optimizer.get('epochs', 10)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            total_correct = 0

            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_correct += (outputs.argmax(1) == targets).sum().item()

            avg_loss = total_loss / len(loader)
            print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {total_correct / len(loader.dataset):.4f}')


    def build_method(self, rebuild=False):
        if rebuild:
            self.train_method()

    def train_method(self):
        pass

    def predict(self, inputs: torch.Tensor):
        """Make predictions in a conventional manner.

        Args:
            inputs: Input tensor

        Returns:
            The predictions of the model
        """
        inputs = inputs.to(self.device)
        self.model.eval()

        predictions = self.model(inputs)
        return predictions

    @abstractmethod
    def measure_uncertainty(self, inputs: torch.Tensor, targets: torch.Tensor):
        """Measure uncertainty. Must be implemented by child classes.

        Returns:
            Dictionary containing uncertainty measures:
                - total_uncertainty
                - aleatoric_uncertainty (data uncertainty)
                - epistemic_uncertainty (model uncertainty)
                - out_of_distribution (OOD score)
        """
        return {}
