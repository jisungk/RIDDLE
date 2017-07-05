"""
roc.py

Provides functions for computing ROC AUC and plotting then saving ROC curves.

Requires:   NumPy, SciPy, matplotlib, scikit-learn (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
"""

from __future__ import print_function

import os

import numpy as np
from scipy import interp
import matplotlib as mpl; mpl.use('Agg') # do not run X server
import matplotlib.pyplot as plt; plt.style.use('ggplot')
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# --------------------------------- SETTINGS --------------------------------- #

np.random.seed(109971161161043253 % 8085) # for reproducibility

# -------------------------------- FUNCTIONS --------------------------------- #

''' 
* Computes ROC_AUC for each class.
* Expects:
    - y_test = list of true class indices
    - y_test_probas = list of predicted probability vectors
    - nb_classes = number of classes
* Returns:
    - dictionary of ROC_AUC values
    - dictinoary of FPR values 
    - dictionary ot TPR values
'''
def compute_roc(y_test, y_test_probas, nb_classes):
    y_test = label_binarize(y_test, classes=range(0, nb_classes))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_test_probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), 
        y_test_probas.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc, fpr, tpr

''' 
* Creates and saves ROC plots to file.
* Expects:
    - roc_auc = dictionary of ROC_AUC values
    - fpr = dictionary of false positive rates
    - tpr = dictionary of true positive rate
    - nb_classes = number of classes
    - path = string filepath describing where to save files
'''
# Plot ROC curves for the multiclass problem
def save_plots(roc_auc, fpr, tpr, nb_classes, path):
    # aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

    # interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # average and compute AUC
    mean_tpr /= nb_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # plot
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], 
        label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
        linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"],
        label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
        linewidth=2)
    for i in range(nb_classes):
        plt.plot(fpr[i], tpr[i], 
            label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")

    plt.savefig(path) # save plot
    plt.close()