import torch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def entropy(id: torch.tensor, ood: dict, kde=True):
    res = pd.DataFrame()
    for k_ in id.keys():
        if isinstance(id[k_], int):
            continue
        res = pd.concat([res, pd.DataFrame({'type': ['ID'] * len(id[k_]),
                                            'uncertainty': [k_] * len(id[k_]),
                                            'Entropy': list(
                                                id[k_])})],
                        ignore_index=True, )
        for k, v in ood.items():
            res = pd.concat(
                [res, pd.DataFrame({'type': [k] * len(v[k_]), 'Entropy': list(v[k_]), 'uncertainty': [k_] * len(v[k_])})],
                ignore_index=True, )
        # Facet grid with one subplot per "dist"
    g = sns.FacetGrid(res, col="uncertainty", hue="type", sharex=False)
    g.map(sns.histplot, "Entropy", stat="probability", element="step", kde=kde)

    return g

def roc(id_scores, ood_scores, plot_title):
    true_labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    uncertainty_scores = np.concatenate([id_scores, ood_scores])  # higher = more OOD-like
    fpr, tpr, _ = roc_curve(true_labels, uncertainty_scores)
    auroc = roc_auc_score(true_labels, uncertainty_scores)
    plt.plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title(f"ID vs OOD classification on {plot_title.replace('_', ' ')}")

    return plt



