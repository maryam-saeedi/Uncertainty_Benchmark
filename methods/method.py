from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import os
from models.model_factory import ModelFactory
from utils.metrics import *
import numpy as np
from torch.nn import functional as F


class Method(ABC):
    """Base method class for all uncertainty quantification methods."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps = 1e-10
        self.num_classes = config.dataset.num_classes
        self.init_model()
        self.init_optimizer()

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

    def save_pretrained_model(self, pretrained):
        print(pretrained)
        torch.save(self.model.state_dict(), pretrained)
        print("--- Model saved ---")

    def load_pretrained_model(self, pretrained):
        self.model.load_state_dict(torch.load(pretrained, weights_only=True, map_location=self.device))

    def build_base_model(self, retrain=False, **kwargs):
        if retrain:
            assert kwargs['train_loader'] is not None and kwargs['val_loader'] is not None, "Train and validation loaders must exist in order to re-train the model."
            print('going for training')
            if self.config.weighted:
                if 'weights' in self.config.dataset:
                    weights = torch.tensor(self.config.dataset.weights).float().to(self.device)
                else:
                    labels = torch.tensor(kwargs['train_loader'].dataset.labels)
                    class_counts = torch.bincount(labels, minlength=self.num_classes)
                    class_counts[class_counts == 0] = 1

                    N = labels.size(0)
                    weights = N / (self.num_classes * class_counts.float())
                    weights = weights.to(self.device)
                print("weights", weights)
                self.train_base_model(kwargs['train_loader'], kwargs['val_loader'], weights)
            else:
                self.train_base_model(kwargs['train_loader'], kwargs['val_loader'])
            print('train is done')
            os.makedirs(self.config.output.base_model_path, exist_ok=True)
            if 'model_name' in kwargs:
                model_name = kwargs['model_name']
            else:
                model_name = f'base_model_{self.config.model.name}.pt'
            print(self.model)
            # torch.save(self.model.state_dict(), os.path.join(self.config.output.base_model_path, model_name))
            self.save_pretrained_model(os.path.join(self.config.output.base_model_path, model_name))
        else:
            assert kwargs['pretrained'] is not None, "Pretrained checkpoint cannot be None to use a pretrained model."
            # self.model.load_state_dict(torch.load(kwargs['pretrained'], weights_only=True))
            self.load_pretrained_model(kwargs['pretrained'])

    def train_base_model(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, loss_weight=None):
        """Train the model using standard supervised learning.
        Args:
            loader: Training data loader
        """
        # Setup optimizer
        optimizer = self.optimizer
        criterion = nn.CrossEntropyLoss(loss_weight)

        # Training

        best_acc = 0.0
        best_previous_loss = float('inf')
        epochs = self.config.optimizer.get('epochs', 10)
        print("Any trainable params:",
              any(p.requires_grad for p in self.model.parameters()))
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_correct = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_correct += (outputs.argmax(1) == targets).sum().item()

            avg_loss = total_loss / len(train_loader)

            # -------- VALIDATION --------
            self.model.eval()
            val_loss = 0.0
            val_correct = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)
                    # print(outputs, targets)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    val_correct += (outputs.argmax(1) == targets).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / len(val_loader.dataset)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), f'best_model.pt')

            print(f'Epoch {epoch+1}/{epochs} - Train: Loss: {avg_loss:.4f}, Accuracy: {total_correct / len(train_loader.dataset):.4f}\nValidation: Loss {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}')

        self.model.load_state_dict(torch.load('best_model.pt'))
        os.remove('best_model.pt')

    def build_method(self, rebuild=False, **kwargs):
        if rebuild:
            model = self.train_uncertainty_method(kwargs['train_loader'], kwargs['valid_loader'])
            if model is not None:
                if 'model_name' in kwargs:
                    model_name = kwargs['model_name']
                else:
                    model_name = f'model.pt'
                torch.save(model, os.path.join(self.config.output.path, self.config.method.name, 'model', model_name))

    def train_uncertainty_method(self, train_loader, valid_loader):
        return None

    def inference(self, loader: torch.utils.data.DataLoader):
        """Make predictions in a conventional manner.
        Args:
            inputs: Input tensor

        Returns:
            The predictions of the model
        """
        self.model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)

                output = self.model(inputs)
                prediction = F.softmax(output, dim=1)
                predictions.append(prediction)
                labels.append(targets)

        predictions = torch.cat(predictions)
        labels = torch.cat(labels)
        return predictions, labels

    def measure_uncertainty(self, loader: torch.utils.data.DataLoader):
        """Measure uncertainty. Must be implemented by child classes.

        Returns:
            Dictionary containing uncertainty measures:
                - total_uncertainty
                - aleatoric_uncertainty (data uncertainty)
                - epistemic_uncertainty (model uncertainty)
                - out_of_distribution (OOD score)
        """
        predictions, ground_truth = self.inference(loader)    # shape [T, B, C]
        print(predictions.shape)
        aleatoric_uncertainty = -torch.mean(torch.sum(predictions * torch.log(predictions+self.eps), dim=2), dim=0)

        p_mean = torch.mean(predictions, dim=0)
        total_uncertainty = -torch.sum(p_mean * torch.log(p_mean+self.eps), dim=1)

        epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

        mi = (predictions * (torch.log(predictions + self.eps) - torch.log(p_mean + self.eps))).sum(dim=2).mean(dim=0)  # [N]

        var_epistemic = predictions.var(dim=0).sum(dim=-1)  # [B, C] --> [B]
        var_aleatoric = (predictions * (1 - predictions)).mean(dim=0).sum(dim=-1)
        var_total = var_epistemic + var_aleatoric

        return {
            "predictions": p_mean,
            "predicted_labels": p_mean.argmax(dim=-1),
            "ground_truth": ground_truth,
            "total_uncertainty": total_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "mutual_information": mi,
            "variance_epistemic_uncertainty": var_epistemic,
            "variance_aleatoric_uncertainty": var_aleatoric,
            "variance_total_uncertainty": var_total,
        }

    def evaluate_method(self, predictions, targets):
        nll = get_NLL_score(predictions, targets)
        accuracy = get_acc_score(predictions, targets)
        auc = get_auc_score(predictions, targets)

        return {
            "nll": nll.detach().cpu().numpy(),
            "accuracy": accuracy.detach().cpu().numpy(),
            "auc": auc
        }
