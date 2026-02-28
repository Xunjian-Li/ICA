# path: faster-ica/ml_ica/algorithms/natural_gradient.py

from __future__ import print_function

from time import time

import numpy as np
from scipy import linalg

from ml_ica.tools import gradient, score


def natural_gradient_ica(
    X,
    max_iter=1000,
    tol=1e-7,
    step_size=0.1,
    verbose=0,
    callback=None,
    sym_decorrelate=False,
    print_every=1,
):
    """
    Natural gradient ICA with Picard-compatible verbose output.

    Key fix:
    - Keep W/I/Y/G in the same dtype as X (float32 stays float32).
    - Avoid implicit float64 upcasting via np.eye, scalar constants, etc.
    """
    X = np.asarray(X, order="F")
    N, T = X.shape
    dtype = X.dtype

    if dtype not in (np.float32, np.float64):
        # be explicit: ICA code expects floating point
        X = X.astype(np.float64, copy=False)
        dtype = X.dtype

    W = np.eye(N, dtype=dtype)
    W_old = np.ones_like(W)
    I = np.eye(N, dtype=dtype)

    step = dtype.type(step_size)
    eps = dtype.type(1e-12)

    Y = W @ X
    t0 = time()

    for n in range(max_iter):
        timing = time() - t0

        psiY = score(Y)
        G = gradient(Y, psiY).astype(dtype, copy=False)

        gradient_norm = linalg.norm(G.ravel(), ord=np.inf)
        diff_W = np.sum(np.abs(W - W_old)) / (N * N)

        if n > 0 and diff_W < tol:
            break

        W_old = W.copy()
        W = (I - step * G) @ W

        if sym_decorrelate:
            # SciPy may compute in float64 internally; cast back to dtype.
            s, u = linalg.eigh((W @ W.T).astype(np.float64, copy=False))
            s = s.astype(dtype, copy=False)
            u = u.astype(dtype, copy=False)

            inv_sqrt = dtype.type(1.0) / np.sqrt(np.maximum(s, eps))
            W = (u @ np.diag(inv_sqrt) @ u.T) @ W

        Y = W @ X

        if verbose and (print_every is None or print_every <= 1 or (n % print_every == 0)):
            info = "iteration %d, gradient norm = %.4g" % (n, gradient_norm)
            ending = "\r" if verbose == 1 else "\n"
            print(info, end=ending)

        if callback is not None:
            callback(locals())

    return Y, W
