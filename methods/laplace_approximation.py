import os

from methods.method import Method
from laplace import Laplace
import torch
import numpy as np
from tqdm import tqdm
import pickle

class MLP:
    pass

class LaplaceApproximation(Method):
    def __init__(self, config):
        super(LaplaceApproximation, self).__init__(config)

    def build_method(self, rebuild=False, **kwargs):
        if not rebuild:
            self.model.train()
            self.laplace = Laplace(self.model,  "classification",
                               self.config.method.subset_of_weights,
                               self.config.method.hessian_structure)

            self.laplace.load_state_dict(torch.load(os.path.join(self.config.method.output.path, 'model', 'laplace.pt'), map_location='cpu'))
            return
        self.train_method(kwargs['train_dl'], kwargs['valid_dl'])
        output_save_dir = os.path.join(self.config.output.path, 'model')
        os.makedirs(output_save_dir, exist_ok=True)
        with open(os.path.join(output_save_dir,'laplace.pkl'), 'wb') as f:
            pickle.dump(self.laplace, f)

    def train_method(self, train_loader, val_loader):

        self.laplace = Laplace(self.model, "classification",
                               self.config.method.subset_of_weights,
                               self.config.method.hessian_structure)

        self.laplace.fit(train_loader)
        self.laplace.optimize_prior_precision(
            method="gridsearch",
            pred_type="glm",
            link_approx="mc",
            val_loader=val_loader
        )

    def measure_uncertainty(self, loader):
        self.model.eval()
        self.laplace.model.eval()
        predictions = []
        for x_test, y_test in tqdm(loader):
            try:
                # User-specified predictive approx.Laplace Redux – Effortless Bayesian Deep Learning
                pred = self.laplace.predictive_samples(x_test.to(self.device), n_samples=1000)
                pred = pred.squeeze(1).unsqueeze(0)
                predictions.append(pred.cpu())
            except Exception as e:
                print(e)
                # pass

        predictions = torch.cat(predictions, dim=0)
        aleatoric_uncertainty = -torch.mean(torch.sum(predictions * torch.log(predictions), dim=2), dim=1)

        p_mean = torch.mean(predictions, dim=1)
        total_uncertainty = -torch.sum(p_mean * torch.log(p_mean), dim=1)

        # Mutual information (BALD) -> epistemic uncertainty
        epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty


        return {
            "total_uncertainty": total_uncertainty.detach().cpu().numpy(),
            "aleatoric_uncertainty": aleatoric_uncertainty.detach().cpu().numpy(),
            "epistemic_uncertainty": epistemic_uncertainty.detach().cpu().numpy(),
            "out_of_distribution": 0,
        }