"""Couplings between Gaussian noise X and data Y within a minibatch.

independent: (X_i, Y_i) as drawn.
ot:          sigma* = argmin_sigma sum_i ||X_i - Y_sigma(i)||^2, exact Hungarian (scipy, float64).
"""
import time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

S = 0.8  # noise level: Z = S*X + (1-S)*Y


def ot_perm(x, y):
    """Returns perm (LongTensor on y.device) with Y[perm] matched to X, plus wall time (s) incl. cost."""
    t0 = time.perf_counter()
    xf, yf = x.flatten(1).double(), y.flatten(1).double()
    cost = torch.cdist(xf, yf).pow(2).cpu().numpy()  # B x B
    r, c = linear_sum_assignment(cost)
    assert np.array_equal(r, np.arange(len(r)))
    perm = torch.as_tensor(c, device=y.device, dtype=torch.long)
    return perm, time.perf_counter() - t0


def ot_perm_within(x, y, labels):
    """Hungarian within each class: noise is only matched to images of the same label (conditional coupling)."""
    t0 = time.perf_counter()
    perm = torch.arange(len(y), device=y.device)
    for k in labels.unique():
        idx = torch.nonzero(labels == k)[:, 0]
        p, _ = ot_perm(x[idx], y[idx])
        perm[idx] = idx[p]
    return perm, time.perf_counter() - t0


def couple(x, y, kind, labels=None):
    """Returns Y re-indexed to pair with X. With labels, OT stays within class (labels are unchanged by it)."""
    if kind == "ind":
        return y, 0.0
    if kind == "ot":
        perm, dt = ot_perm(x, y) if labels is None else ot_perm_within(x, y, labels)
        return y[perm], dt
    raise ValueError(kind)


def interp(x, y, s=S):
    """Z = s X + (1-s) Y. s=1: raw noise input (D regresses Y_pi on X directly)."""
    return s * x + (1 - s) * y
