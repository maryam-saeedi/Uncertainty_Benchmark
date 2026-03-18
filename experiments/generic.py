import pickle

import numpy as np
import pandas as pd
import torch

from methods.method_factory import MethodFactory
from methods.test_time_augmentation import TTA
from load_data import *
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import os
from utils.visualization import *
from utils.id_ood_classification import *
import matplotlib.pyplot as plt

print(pickle.format_version)


train_dl, val_dl, test_dl = load_data_2('/home/msafa/PhD/morpho/Morpho-MNIST/data')
print("data has been loaded,", test_dl.keys())

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config):
    # Print the configuration
    print(config)
    print("Configuration loaded:")
    print(OmegaConf.to_yaml(config))

    retrain_base_model = False
    method = MethodFactory.create(config)

    os.makedirs(os.path.join(config.output.path, config.method.name), exist_ok=True)
    os.makedirs(os.path.join(config.output.path, config.method.name, 'model'), exist_ok=True)
    os.makedirs(os.path.join(config.output.path, config.method.name, 'result'), exist_ok=True)
    os.makedirs(os.path.join(config.output.path, config.method.name, 'plots'), exist_ok=True)

    if not retrain_base_model and os.path.exists(
            os.path.join(config.output.base_model_path, f"base_model_{config.model.name}.pt")):
        method.build_base_model(retrain=False, pretrained=os.path.join(config.output.base_model_path,
                                                                       f"base_model_{config.model.name}.pt"),
                                train_loader=train_dl, val_loader=val_dl)
    else:
        method.build_base_model(retrain=True, train_loader=train_dl, val_loader=val_dl,
                                model_name=f"base_model_{config.model.name}.pt")
    os.makedirs(os.path.join(config.output.path, config.method.name, 'model'), exist_ok=True)
    if os.path.exists(os.path.join(config.output.path, config.method.name, 'model', f"model.pt")):
        method.build_method(rebuild=False, train_loader=train_dl, valid_loader=val_dl,
                            pretrained=os.path.join(config.output.path, config.method.name, 'model',
                                                    f"model.pt"))
    else:
        method.build_method(rebuild=True, train_loader=train_dl, valid_loader=val_dl,
                            model_name=f"model.pt")

    #
    if os.path.exists(os.path.join(config.output.path, config.method.name, 'result', f"id_uncertainty.pkl")):
        with open(os.path.join(config.output.path, config.method.name, 'result', f"id_uncertainty.pkl"), 'rb') as f:
            id_uncertainty = pickle.load(f)
    else:
        id_uncertainty = method.measure_uncertainty(val_dl)
        with open(os.path.join(config.output.path, config.method.name, 'result', f"id_uncertainty.pkl"), 'wb') as f:
            pickle.dump(id_uncertainty, f)

    ood_uncertainties = {}
    for k, v in test_dl.items():
        all_uncertainties = {}
        for k_, v_ in v.items():
            print(k, k_)
            if os.path.exists(os.path.join(config.output.path, config.method.name, 'result', f'testset_uncertainty_{k}_{k_}.pkl')):
                with open(os.path.join(config.output.path, config.method.name, 'result', f'testset_uncertainty_{k}_{k_}.pkl'), 'rb') as f:
                    uncertainties = pickle.load(f)
            else:
                uncertainties = method.measure_uncertainty(v_)
                print([uncertainties[key].shape for key in uncertainties.keys()])
                with open(os.path.join(config.output.path, config.method.name, 'result', f'testset_uncertainty_{k}_{k_}.pkl'), 'wb') as f:
                    pickle.dump(uncertainties, f)
            all_uncertainties[k_] = uncertainties

        ood_uncertainties[k] = {config.method.name:all_uncertainties}


    plot = moving_out_of_distribution_compare(ood_uncertainties, uncertainty_type="epistemic_uncertainty")
    plot.savefig(os.path.join(config.output.path, config.method.name, 'plots', 'moving_out_of_dist_epistemic.png'))
    plt.close()


    plot = moving_out_of_distribution_compare(ood_uncertainties, uncertainty_type="aleatoric_uncertainty")
    plot.savefig(os.path.join(config.output.path, config.method.name, 'plots', 'moving_out_of_dist_aleatoric.png'))
    plt.close()

    plot = moving_out_of_distribution_compare(ood_uncertainties, uncertainty_type="total_uncertainty")
    plot.savefig(os.path.join(config.output.path, config.method.name, 'plots', 'moving_out_of_dist_total.png'))
    plt.close()


    # plot = ood_roc(ood_uncertainties, id_uncertainty)
    # plot.savefig(os.path.join(config.output.path, config.method.name, 'plots', f"ood_auroc.png"))
    # plt.close()

if __name__ == "__main__":
        main()

