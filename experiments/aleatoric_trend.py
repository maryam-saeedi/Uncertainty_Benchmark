import pickle
import sys

import numpy as np
import pandas as pd

from load_data import *
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from utils.visualization import *
from utils.id_ood_classification import *
import matplotlib.pyplot as plt
from methods import MethodFactory
from glob import glob


def set_random_seed(seed):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["CUDNN_DETERMINISTIC"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

train_dl, val_dl = load_aleatoric_data3('/home/msafa/PhD/morpho/Morpho-MNIST/data')

@hydra.main(config_path="../configs", config_name="config_aleatoric_trend", version_base=None)
def main(config: DictConfig) -> None:
    # Print the configuration
    print("Configuration loaded:")
    print(OmegaConf.to_yaml(config))
    set_random_seed(config.seed)

    method = MethodFactory.create(config)
    print(train_dl.keys())

    retrain = True
    uncertainties = {}
    for name in train_dl.keys():
        if os.path.exists(os.path.join(config.output.base_model_path, f"base_model_{config.model.name}_{name}.pt")):
            method.build_base_model(retrain=False, pretrained=os.path.join(config.output.base_model_path,
                                                                           f"base_model_{config.model.name}_{name}.pt"))
        else:
            method.build_base_model(retrain=True, train_loader=train_dl[name], val_loader=val_dl[name],
                                    model_name=f"base_model_{config.model.name}_{name}.pt")
        os.makedirs(os.path.join(config.output.path, config.method.name, 'model'), exist_ok=True)
        if os.path.exists(os.path.join(config.output.path, config.method.name, 'model', f"model_{name}.pt")):
            method.build_method(rebuild=False, train_dl=train_dl[name],
                                pretrained=os.path.join(config.output.path, config.method.name, 'model',
                                                        f"model_{name}.pt"))
        else:
            method.build_method(rebuild=True, train_loader=train_dl[name], valid_loader=val_dl[name],
                                model_name=f"model_{name}.pt")

        ## inference
        os.makedirs(os.path.join(config.output.path, config.method.name, 'result'), exist_ok=True)

        if os.path.exists(os.path.join(config.output.path, config.method.name, 'result', f'validset_uncertainty_{name}.pkl')):
            with open(os.path.join(config.output.path, config.method.name, 'result', f'validset_uncertainty_{name}.pkl'), 'rb') as f:
                uncertainty = pickle.load(f)
        else:
            uncertainty = method.measure_uncertainty(val_dl[name])
            with open(os.path.join(config.output.path, config.method.name, 'result', f'validset_uncertainty_{name}.pkl'), 'wb') as f:
                pickle.dump(uncertainty, f)
        uncertainties[name] = uncertainty

    plot = trend(uncertainties)
    os.makedirs(os.path.join(config.output.path, config.method.name, 'plots'), exist_ok=True)
    plot.savefig(os.path.join(config.output.path, config.method.name, 'plots', 'trend.png'))
    plt.close()


if __name__ == "__main__":
    print(sys.argv[1])
    main()

