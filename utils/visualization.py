import torch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def entropy(id: torch.tensor, ood: dict, kde=True):
    res = pd.DataFrame()
    if id is not None:
        for k_ in ['total_uncertainty', 'aleatoric_uncertainty', 'epistemic_uncertainty']:
            res = pd.concat([res, pd.DataFrame({'type': ['ID'] * len(id[k_]),
                                        'uncertainty': [k_] * len(id[k_]),
                                        'Entropy': list(
                                            id[k_])})],
                    ignore_index=True, )
    for k, v in ood.items():
        for k_ in ['total_uncertainty', 'aleatoric_uncertainty', 'epistemic_uncertainty']:
            res = pd.concat(
                [res, pd.DataFrame({'type': [k] * len(v[k_]), 'Entropy': list(v[k_]), 'uncertainty': [k_] * len(v[k_])})],
                ignore_index=True, )
    g = sns.FacetGrid(res, col="uncertainty", hue="type", sharex=False)
    g.map(sns.histplot, "Entropy", stat="probability", element="step", kde=kde)
    g.add_legend()
    return g

def roc(id_scores, ood_scores):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    for k in id_scores.keys():
        for i, k_ in enumerate(['total_uncertainty', 'aleatoric_uncertainty', 'epistemic_uncertainty']):
            true_labels = np.concatenate([np.zeros_like(id_scores[k][k_]), np.ones_like(ood_scores[k][k_])])
            uncertainty_scores = np.concatenate([id_scores[k][k_], ood_scores[k][k_]])  # higher = more OOD-like
            fpr, tpr, _ = roc_curve(true_labels, uncertainty_scores)
            auroc = roc_auc_score(true_labels, uncertainty_scores)
            axs[i].plot(fpr, tpr, label=f"{k}_AUROC={auroc:.3f}")
            axs[i].plot([0,1],[0,1],'--', color='gray')
    plt.legend()

    return plt

def error_rate(uncertainty_scores, predictions, true_labels, kde=True):
    res = pd.DataFrame()
    predicted_labels = torch.argmax(predictions, dim=-1).numpy()
    for k_ in ['total_uncertainty', 'aleatoric_uncertainty', 'epistemic_uncertainty']:
        print(predicted_labels.shape)
        uncertainties = uncertainty_scores[k_]
        print(uncertainties.shape)
        errors = np.sum((predicted_labels != np.expand_dims(true_labels,1)), axis=1)
        res = pd.concat([res, pd.DataFrame({'Error Rate':errors, 'Uncertainty Score': uncertainties, 'uncertainty': [k_]*len(uncertainties)})], ignore_index=True)

    # Create FacetGrid
    g = sns.FacetGrid(res, col="uncertainty", sharex=False, sharey=False)

    # Plot 2D histogram
    g.map_dataframe(sns.histplot, x="Uncertainty Score", y="Error Rate", stat="probability", bins=100)

    # Overlay mean line per facet
    def mean_line(data, **kwargs):
        # Bin x values
        bins = np.linspace(data["Uncertainty Score"].min(), data["Uncertainty Score"].max(), 30)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        means = [data.loc[(data["Uncertainty Score"] >= bins[i]) & (
                    data["Uncertainty Score"] < bins[i + 1]), "Error Rate"].mean()
                 for i in range(len(bins) - 1)]
        plt.plot(bin_centers, means, color='red', linewidth=2)

    g.map_dataframe(mean_line)
    g.add_legend()

    return g

def mean_error_rate(uncertainty_scores, mean_predictions, true_labels, kde=True):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for i, k_ in enumerate(['total_uncertainty', 'aleatoric_uncertainty', 'epistemic_uncertainty']):
        res = pd.DataFrame()
        for k in uncertainty_scores.keys():
            predicted_labels = torch.argmax(mean_predictions[k], dim=-1).numpy()
            scores = uncertainty_scores[k][k_]
            df = pd.DataFrame({'Uncertainty Score': scores, 'Error Rate':predicted_labels!=true_labels})

            bins = np.linspace(df["Uncertainty Score"].min(), df["Uncertainty Score"].max(), 50)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            means = [df.loc[(df["Uncertainty Score"] >= bins[i]) & (
                    df["Uncertainty Score"] < bins[i + 1]), "Error Rate"].mean()
                     for i in range(len(bins) - 1)]

            res = pd.concat([res, pd.DataFrame({'Uncertainty Score': bin_centers, 'Error Rate': means,'type':[k]*len(bin_centers), 'uncertainty': [k_]*len(bin_centers)})], ignore_index=True)

        sns.lineplot(res, x='Uncertainty Score', y='Error Rate', hue='type', ax=axs[i])
        axs[i].set_title(f"{k_}")
    return plt
