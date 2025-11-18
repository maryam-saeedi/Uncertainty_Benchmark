import pickle

from methods.laplace_approximation import LaplaceApproximation
from load_data import load_data
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from utils.visualization import *
from utils.id_ood_classification import *
import matplotlib.pyplot as plt


train_dl, val_dl, test_dl = load_data()

@hydra.main(config_path="../configs", config_name="la1", version_base=None)
def main(config: DictConfig) -> None:
    # Print the configuration
    print("Configuration loaded:")
    print(OmegaConf.to_yaml(config))

    la = LaplaceApproximation(config)
    la.build_base_model(retrain=False, pretrained='results/common/model.pt')
    la.build_method(rebuild=False)


    if os.path.exists(os.path.join(config.method.output.path, 'result', 'valid_uncertainties.pkl')):
        with open(os.path.join(config.method.output.path, 'result', 'valid_uncertainties.pkl'), 'rb') as f:
            valid_uncertainties = pickle.load(f)
    else:
        valid_uncertainties = la.measure_uncertainty(val_dl)
        with open(os.path.join(config.method.output.path, 'result', 'valid_uncertainties.pkl'), 'wb') as f:
            pickle.dump(valid_uncertainties, f)

    ood_uncertainties = {}
    for k, v in test_dl.items():
        if os.path.exists(os.path.join(config.method.output.path, 'result', f'test_uncertainties_{k}.pkl')):
            with open(os.path.join(config.method.output.path, 'result', f'test_uncertainties_{k}.pkl'), 'rb') as f:
                test_uncertainties = pickle.load(f)
        else:
            test_uncertainties = la.measure_uncertainty(v)
            with open(os.path.join(config.method.output.path, 'result', f'test_uncertainties_{k}.pkl'), 'wb') as f:
                pickle.dump(test_uncertainties, f)
        ood_uncertainties[k] = test_uncertainties

    plot = entropy(valid_uncertainties, ood_uncertainties)
    os.makedirs(os.path.join(config.method.output.path, 'plots'), exist_ok=True)
    plot.savefig(os.path.join(config.method.output.path, 'plots', 'entropy.png'))
    plt.close(plot.fig)

    for k_ in ['total_uncertainty', 'aleatoric_uncertainty', 'epistemic_uncertainty']:
        for k, v in test_dl.items():
            roc_plot = roc(id_scores=valid_uncertainties[k_], ood_scores=ood_uncertainties[k][k_], plot_title=k_)
            roc_plot.savefig(os.path.join(config.method.output.path, 'plots', f'{k_}_{k}_roc.png'))
            plt.close()


if __name__ == "__main__":
    main()

