import pickle

import numpy as np
import pandas as pd

from methods.method_factory import MethodFactory
from load_data import *
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from utils.visualization import *
import matplotlib.pyplot as plt


train_dl, val_dl, test_dl = load_data_different_training_size('/home/msafa/PhD/morpho/Morpho-MNIST/data')

@hydra.main(config_path="../configs", config_name="config_epistemic_trend", version_base=None)
def main(config: DictConfig) -> None:
    # Print the configuration
    print("Configuration loaded:")
    print(OmegaConf.to_yaml(config))

    method = MethodFactory.create(config)
    print(train_dl.keys())

    retrain_base_model = False

    uncertainties = {}
    for name in train_dl.keys():
        print(f"base_model_{config.model.name}_{name}.pt")
        if not retrain_base_model and os.path.exists(os.path.join(config.output.base_model_path, f"base_model_{config.model.name}_{name}.pt")):
            method.build_base_model(retrain=False, pretrained=os.path.join(config.output.base_model_path,
                                                                           f"base_model_{config.model.name}_{name}.pt"))
        else:
            method.build_base_model(retrain=True, train_loader=train_dl[name], val_loader=val_dl,
                                    model_name=f"base_model_{config.model.name}_{name}.pt")
        os.makedirs(os.path.join(config.output.path, config.method.name, 'model'), exist_ok=True)
        if os.path.exists(os.path.join(config.output.path, config.method.name, 'model', f"model_{name}.pt")):
            method.build_method(rebuild=False, train_loader=train_dl[name],
                                pretrained=os.path.join(config.output.path, config.method.name, 'model',
                                                        f"model_{name}.pt"))
        else:
            method.build_method(rebuild=True, train_loader=train_dl[name], valid_loader=val_dl,
                                model_name=f"model_{name}.pt")

        ## inference
        os.makedirs(os.path.join(config.output.path, config.method.name, 'result'), exist_ok=True)

        if os.path.exists(os.path.join(config.output.path, config.method.name, 'result', f'validset_uncertainty_{name}.pkl')):
            with open(os.path.join(config.output.path, config.method.name, 'result', f'validset_uncertainty_{name}.pkl'), 'rb') as f:
                uncertainty = pickle.load(f)
        else:
            uncertainty = method.measure_uncertainty(test_dl)
            with open(os.path.join(config.output.path, config.method.name, 'result', f'validset_uncertainty_{name}.pkl'), 'wb') as f:
                pickle.dump(uncertainty, f)
        uncertainties[name] = uncertainty

    plot = trend(uncertainties)
    os.makedirs(os.path.join(config.output.path, config.method.name, 'plots'), exist_ok=True)
    plot.savefig(os.path.join(config.output.path, config.method.name, 'plots', 'trend.png'))
    plt.close()


if __name__ == "__main__":
    main()

