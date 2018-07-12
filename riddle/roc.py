"""roc.py

Provides functions for computing ROC AUC and plotting then saving ROC curves
in a multiclass setting.

Requires:   NumPy, SciPy, matplotlib, scikit-learn (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2018, all rights reserved
"""

from __future__ import print_function

import numpy as np
import matplotlib
matplotlib.use('Agg')  # do not run X server

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def compute_and_plot_roc_scores(y_test, y_test_probas, num_class, path=None):
    """Compute ROC statistics and plot ROC curves.

    Arguments:
        y_test: [int]
            list of test class labels as integer indices
        y_test_probas: np.ndarray, float
            array of predicted probabilities with shape
            (num_sample, num_class)
        num_class: int
            number of classes
        path: string
            filepath where to save the ROC curve plot; if None will not perform
            plotting

    Returns:
        roc_auc_dict: {int: float}
            dictionary mapping classes to ROC AUC scores
        fpr_dict: {string: np.ndarray}
            dictionary mapping names of classes or an averaging method to
            arrays of increasing false positive rates
        tpr_dict: {string: float}
            dictionary mapping names of classes or an averaging method to
            arrays of increasing true positive rates
    """
    roc_auc_dict, fpr_dict, tpr_dict = _compute_roc_stats(y_test, y_test_probas,
                                                          num_class)

    if path is not None:
        _create_roc_plot(roc_auc_dict, fpr_dict, tpr_dict, num_class, path)

    return roc_auc_dict, fpr_dict, tpr_dict


def _create_roc_plot(roc_auc_dict, fpr_dict, tpr_dict, num_class, path):
    """Create and save a combined ROC plot to file.

    Arguments:
        roc_auc_dict: {int: float}
            dictionary mapping classes to ROC AUC scores
        fpr_dict: {string: np.ndarray}
            dictionary mapping names of classes or an averaging method to
            arrays of increasing false positive rates
        tpr_dict: {string: float}
            dictionary mapping names of classes or an averaging method to
            arrays of increasing true positive rates
        num_class: int
            number of classes
        path: string
            filepath where to save the plot
    """
    # aggregate all false positive rates
    all_fpr = np.unique(np.concatenate(
        [fpr_dict[i] for i in range(num_class)]))

    # interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])

    # average and compute AUC
    mean_tpr /= num_class

    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

    # plot
    plt.figure()
    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'.format(
                 roc_auc_dict["micro"]),
             linewidth=2)
    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'.format(
                 roc_auc_dict["macro"]),
             linewidth=2)
    for i in range(num_class):
        plt.plot(fpr_dict[i], tpr_dict[i],
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(
                     i, roc_auc_dict[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")

    plt.savefig(path)  # save plot
    plt.close()


def _compute_roc_stats(y_test, y_test_probas, num_class):
    """Compute ROC AUC statistics and visualize ROC curves.

    Arguments:
        y_test: [int]
            list of test class labels as integer indices
        y_test_probas: np.ndarray, float
            array of predicted probabilities with shape
            (num_sample, num_class)
        num_class: int
            number of classes

    Returns:
        roc_auc_dict: {int: float}
            dictionary mapping classes to ROC AUC scores
        fpr_dict: {string: np.ndarray}
            dictionary mapping names of classes or an averaging method to
            arrays of increasing false positive rates
        tpr_dict: {string: float}
            dictionary mapping names of classes or an averaging method to
            arrays of increasing true positive rates
    """
    y_test = label_binarize(y_test, classes=range(0, num_class))

    fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
    for i in range(num_class):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(
            y_test[:, i], y_test_probas[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

    # Compute micro-average ROC curve and ROC area
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(
        y_test.ravel(), y_test_probas.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    return roc_auc_dict, fpr_dict, tpr_dict
