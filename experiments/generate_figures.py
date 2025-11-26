import numpy as np

from load_data import *
import matplotlib.pyplot as plt
from utils.visualization import *
import pickle
import os

def all_entopies(data_sets=None, methods=None, result_path=''):
    '''
    generate all possible entropy figures
    :return:
    '''
    for method in methods:
        all_data = {}
        for data_type in data_sets:
            ## load corresponding data
            data = pickle.load(open(os.path.join(result_path, method, 'result', f'test_uncertainties_{data_type}.pkl'), "rb"))
            all_data[data_type] = data
        id_data = pickle.load(open(os.path.join(result_path, method, 'result', 'valid_uncertainties.pkl'), "rb"))
        plot = entropy(id_data, all_data)
        plot.savefig(os.path.join(result_path, 'common', 'plots', f'entropy_{method}.png'))
        plt.close(plot.fig)

    for data_type in data_sets:
        all_data = {}
        for method in methods:
            ## load corresponding data
            data = pickle.load(
                open(os.path.join(result_path, method, 'result', f'test_uncertainties_{data_type}.pkl'), "rb"))
            all_data[method] = data
        id_data = pickle.load(open(os.path.join(result_path, method, 'result', 'valid_uncertainties.pkl'), "rb"))
        plot = entropy(id_data, all_data)
        plot.savefig(os.path.join(result_path, 'common', 'plots',f'entropy_{data_type}.png'))
        plt.close(plot.fig)

def all_roc(data_sets=None, methods=None, result_path=''):
    '''
    generate all possible roc figures
    :return:
    '''
    for method in methods:
        id_data = {}
        ood_data = {}
        for data_type in data_sets:
            ## load corresponding data
            data = pickle.load(open(os.path.join(result_path, method, 'result', f'test_uncertainties_{data_type}.pkl'), "rb"))
            ood_data[data_type] = data
            data = pickle.load(open(os.path.join(result_path, method, 'result', 'valid_uncertainties.pkl'), "rb"))
            id_data[data_type] = data
        plot = roc(id_data, ood_data)
        plot.savefig(os.path.join(result_path, 'common', 'plots', f'roc_{method}.png'))
        plt.close()

    for data_type in data_sets:
        id_data = {}
        ood_data = {}
        for method in methods:
            ## load corresponding data
            data = pickle.load(
                open(os.path.join(result_path, method, 'result', f'test_uncertainties_{data_type}.pkl'), "rb"))
            ood_data[method] = data
            data = pickle.load(open(os.path.join(result_path, method, 'result', 'valid_uncertainties.pkl'), "rb"))
            id_data[method] = data
        plot = roc(id_data, ood_data)
        plot.savefig(os.path.join(result_path, 'common', 'plots',f'roc_{data_type}.png'))
        plt.close()

def all_error_rates(data_sets=None, methods=None, val_dl=None, result_path=''):
    for data_type in data_sets:
        uncertainties = {}
        validset_predictions = {}
        for method in methods:
            ## load corresponding data
            data = pickle.load(
                open(os.path.join(result_path, method, 'result', f'validset_uncertainty_{data_type}.pkl'), "rb"))
            uncertainties[method] = data
            data = pickle.load(
                open(os.path.join(result_path, method, 'result', f'validset_predictions_{data_type}.pkl'), "rb"))
            data = torch.mean(data, dim=1)
            validset_predictions[method] = data
        plot = mean_error_rate(uncertainties, validset_predictions, np.concatenate([d.labels for d in val_dl[data_type].dataset.datasets]))
        plot.savefig(os.path.join(result_path, 'common', 'plots', f'error_rate_{data_type}.png'))
        plt.close()

    for method in methods:
        uncertainties = {}
        validset_predictions = {}
        for data_type in data_sets:
            ## load corresponding data
            data = pickle.load(
                open(os.path.join(result_path, method, 'result', f'validset_uncertainty_{data_type}.pkl'), "rb"))
            uncertainties[data_type] = data
            data = pickle.load(
                open(os.path.join(result_path, method, 'result', f'validset_predictions_{data_type}.pkl'), "rb"))
            data = torch.mean(data, dim=1)
            validset_predictions[data_type] = data
        plot = mean_error_rate(uncertainties, validset_predictions, np.concatenate([d.labels for d in val_dl[data_type].dataset.datasets]))
        plot.savefig(os.path.join(result_path, 'common', 'plots', f'error_rate_{method}.png'))
        plt.close()

if __name__ == '__main__':
    methods = ['TTA', 'swag']
    data_sets = ['thining 5', 'thining 9']
    train_dl, val_dl = load_aleatoric_data('/home/msafa/PhD/morpho/Morpho-MNIST/data')
    # all_entopies(data_sets=data_sets, methods=methods, result_path='morpho_mnist_result')
    # all_roc(data_sets=data_sets, methods=methods, result_path='morpho_mnist_result')
    all_error_rates(data_sets=val_dl.keys(), methods=methods, val_dl=val_dl, result_path='blur_mnist_result')