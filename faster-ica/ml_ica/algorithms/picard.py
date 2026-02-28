# # Authors: Pierre Ablin <pierre.ablin@inria.fr>
# #          Alexandre Gramfort <alexandre.gramfort@inria.fr>
# #          Jean-Francois Cardoso <cardoso@iap.fr>
# #
# # License: BSD (3-clause)

# path: faster-ica/ml_ica/algorithms/picard.py

from __future__ import print_function

from time import time

import numpy as np
from scipy import linalg

from ml_ica.tools import (
    gradient, compute_h, regularize_h, solveh,
    score, score_der, linesearch
)


def picard(
    X,
    max_iter=1000,
    tol=1e-7,
    mem_size=7,
    precon=2,
    lambda_min=0.01,
    ls_tries=10,
    verbose=0,
    callback=None,
):
    """
    Runs Picard algorithm.

    Precision support
    -----------------
    This version preserves the dtype of `X` (float32 stays float32, float64 stays float64)
    by:
      - initializing W with dtype=X.dtype
      - ensuring Y/W/G/direction memory are in the same dtype where applicable
      - keeping scalar constants dtype-aware where it matters

    Notes
    -----
    SciPy linear algebra may internally upcast to float64 for some routines.
    We cast outputs back to the input dtype to avoid mixed-dtype slow paths.
    """
    X = np.asarray(X, order="F")
    dtype = X.dtype
    if dtype not in (np.float32, np.float64):
        X = X.astype(np.float64, copy=False)
        dtype = X.dtype

    # Init
    N, T = X.shape
    W = np.eye(N, dtype=dtype)
    W_old = np.ones_like(W)
    Y = X.copy(order="F")

    s_list, y_list, r_list = [], [], []
    current_loss = None
    t0 = time()

    for n in range(max_iter):
        timing = time() - t0

        psiY = score(Y)
        psidY = score_der(psiY)

        G = gradient(Y, psiY).astype(dtype, copy=False)

        gradient_norm = linalg.norm(G.ravel(), ord=np.inf)
        diff_W = np.sum(np.abs(W - W_old)) / (N * N)
        if n > 0 and diff_W < tol:
            break

        W_old = W.copy()

        # Update the memory
        if n > 0:
            s_list.append(direction)  # direction from previous iteration
            y = (G - G_old).astype(dtype, copy=False)
            y_list.append(y)
            denom = np.dot(direction.ravel(), y.ravel())
            r_list.append(dtype.type(1.0) / denom)
            if len(s_list) > mem_size:
                s_list.pop(0)
                y_list.pop(0)
                r_list.pop(0)
        G_old = G

        # Hessian approx + regularization (may upcast internally; cast back)
        h = compute_h(Y, psidY, precon)
        h = regularize_h(h, lambda_min)
        if isinstance(h, np.ndarray) and h.dtype != dtype:
            h = h.astype(dtype, copy=False)

        # L-BFGS direction
        direction = _l_bfgs_direction(G, h, s_list, y_list, r_list, precon, lambda_min)
        if isinstance(direction, np.ndarray) and direction.dtype != dtype:
            direction = direction.astype(dtype, copy=False)

        # line search (may return float64 arrays; cast back)
        converged, new_Y, new_W, new_loss, alpha = linesearch(
            Y, W, direction, current_loss, ls_tries
        )
        if not converged:
            direction = (-G).astype(dtype, copy=False)
            s_list, y_list, r_list = [], [], []
            _, new_Y, new_W, new_loss, alpha = linesearch(
                Y, W, direction, current_loss, ls_tries
            )

        # Apply step (ensure dtype consistency)
        alpha = dtype.type(alpha)
        direction *= alpha

        if isinstance(new_Y, np.ndarray) and new_Y.dtype != dtype:
            new_Y = new_Y.astype(dtype, copy=False)
        if isinstance(new_W, np.ndarray) and new_W.dtype != dtype:
            new_W = new_W.astype(dtype, copy=False)

        Y = new_Y
        W = new_W
        current_loss = new_loss

        if verbose:
            info = "iteration %d, gradient norm = %.4g" % (n, gradient_norm)
            ending = "\r" if verbose == 1 else "\n"
            print(info, end=ending)

        if callback is not None:
            callback(locals())

    return Y, W


def _l_bfgs_direction(G, h, s_list, y_list, r_list, precon, lambda_min):
    dtype = G.dtype
    q = G.copy()
    a_list = []
    for s, y, r in zip(reversed(s_list), reversed(y_list), reversed(r_list)):
        alpha = r * np.sum(s * q)
        a_list.append(alpha)
        q -= alpha * y
    z = solveh(q, h)
    if isinstance(z, np.ndarray) and z.dtype != dtype:
        z = z.astype(dtype, copy=False)
    for s, y, r, alpha in zip(s_list, y_list, r_list, reversed(a_list)):
        beta = r * np.sum(y * z)
        z += (alpha - beta) * s
    return -z
