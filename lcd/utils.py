import numpy as np


def precision_recall_curve(pred, truth):
    indices = np.argsort(pred, kind="mergesort")
    pred = pred[indices]
    truth = truth[indices]
    indices = np.diff(pred).nonzero()[0]
    indices = np.r_[indices, truth.size - 1]
    tp = np.cumsum(truth)[indices]
    fp = np.cumsum(1 - truth)[indices]
    p = tp / (tp + fp)
    p[np.isnan(p)] = 0
    r = tp / tp[-1]
    fpr = fp / fp[-1]
    return p, r
