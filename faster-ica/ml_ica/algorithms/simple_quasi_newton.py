# """
# Python implementation of the simple quasi_newton ICA algorithm.
# Reference:
# M. Zibulevsky, "Blind source separation with relative newton method"
# """

# # Authors: Pierre Ablin <pierre.ablin@inria.fr>
# #          Alexandre Gramfort <alexandre.gramfort@inria.fr>
# #          Jean-Francois Cardoso <cardoso@iap.fr>
# #
# # License: BSD (3-clause)

# from __future__ import print_function
# from time import time
# import numpy as np
# from scipy import linalg

# from ml_ica.tools import (loss, gradient, compute_h, regularize_h, solveh,
#                           score, score_der, linesearch)


# def simple_quasi_newton_ica(X, max_iter=200, tol=1e-7, precon=2,
#                             lambda_min=0.01, ls_tries=10, verbose=0,
#                             callback=None):
#     '''
#     Simple quasi-Newton algorithm.
#     Highly inspired by:

#     M. Zibulevsky, "Blind source separation with relative newton method"

#     Parameters
#     ----------
#     X : array, shape (N, T)
#         Matrix containing the signals that have to be unmixed. N is the
#         number of signals, T is the number of samples. X has to be centered

#     max_iter : int
#         Maximal number of iterations for the algorithm

#     tol : float
#         tolerance for the stopping criterion. Iterations stop when the norm
#         of the gradient gets smaller than tol.

#     precon : 1 or 2
#         Chooses which Hessian approximation is used.
#         1 -> H1
#         2 -> H2
#         H2 is more costly to compute but can greatly accelerate convergence
#         (See the paper for details).

#     lambda_min : float
#         Constant used to regularize the Hessian approximations. The
#         eigenvalues of the approximation that are below lambda_min are
#         shifted to lambda_min.

#     ls_tries : int
#         Number of tries allowed for the backtracking line-search. When that
#         number is exceeded, the direction is thrown away and the gradient
#         is used instead.

#     verbose : 0, 1 or 2
#         Verbose level. 0: No verbose. 1: One line verbose. 2: Detailed verbose

#     callback : None or function
#         Optional function run at each iteration on all the local variables.

#     Returns
#     -------
#     Y : array, shape (N, T)
#         The estimated source matrix

#     W : array, shape (N, N)
#         The estimated unmixing matrix, such that Y = WX.
#     '''
#     Y = X.copy()
#     N, T = Y.shape
#     W = np.eye(N)
#     W_old = np.ones_like(W)
#     current_loss = loss(Y, W)
#     t0 = time()
#     for n in range(max_iter):
#         timing = time() - t0
#         # Compute the score and its derivative
#         psiY = score(Y)
#         psidY = score_der(psiY)
#         # Compute gradient
#         G = gradient(Y, psiY)
#         # Stopping criterion
#         gradient_norm = linalg.norm(G.ravel(), ord=np.inf)
#         diff_W = np.sum(np.abs(W - W_old)) / (N * N)
#         if n > 0 and diff_W < tol:
#             break
#         W_old = W.copy()
#         # Compute the approximation
#         H = compute_h(Y, psidY, precon)
#         # Regularize H
#         H = regularize_h(H, lambda_min)
#         # Compute the descent direction
#         direction = - solveh(G, H)
#         # Do a line_search in that direction
#         success, new_Y, new_W, new_loss, _ =\
#             linesearch(Y, W, direction, current_loss, ls_tries)
#         # If the line search failed, fall back to the gradient
#         if not success:
#             direction = - G
#             _, new_Y, new_W, new_loss, _ =\
#                 linesearch(Y, W, direction, current_loss, ls_tries)
#         # Update
#         Y = new_Y
#         W = new_W
#         current_loss = new_loss
#         # Verbose and callback
#         if callback is not None:
#             callback(locals())
#         if verbose:
#             info = 'iteration %d, gradient norm = %.4g' % (n, gradient_norm)
#             ending = '\r' if verbose == 1 else '\n'
#             print(info, end=ending)
#     return Y, W


# if __name__ == '__main__':
#     N, T = 10, 10000
#     rng = np.random.RandomState(1)
#     S = rng.laplace(size=(N, T))
#     A = rng.randn(N, N)
#     X = np.dot(A, S)
#     simple_quasi_newton_ica(X, verbose=True)


# path: faster-ica/ml_ica/algorithms/simple_quasi_newton.py

"""
Python implementation of the simple quasi_newton ICA algorithm.
Reference:
M. Zibulevsky, "Blind source separation with relative newton method"
"""

from __future__ import print_function

from time import time

import numpy as np
from scipy import linalg

from ml_ica.tools import (
    loss, gradient, compute_h, regularize_h, solveh,
    score, score_der, linesearch,
)


def simple_quasi_newton_ica(
    X,
    max_iter=200,
    tol=1e-7,
    precon=2,
    lambda_min=0.01,
    ls_tries=10,
    verbose=0,
    callback=None,
):
    """
    Simple quasi-Newton ICA.

    Precision support
    -----------------
    This version preserves the dtype of `X` (float32 stays float32, float64 stays float64)
    by:
      - initializing W with dtype=X.dtype
      - keeping Y/W/G/H/direction in the same dtype where possible
      - avoiding implicit float64 upcasts via np.eye and scalar constants
      - casting outputs of helper routines back to dtype if they upcast internally
    """
    X = np.asarray(X, order="F")
    dtype = X.dtype
    if dtype not in (np.float32, np.float64):
        X = X.astype(np.float64, copy=False)
        dtype = X.dtype

    Y = X.copy(order="F")
    N, T = Y.shape

    W = np.eye(N, dtype=dtype)
    W_old = np.ones_like(W)

    current_loss = loss(Y, W)
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

        H = compute_h(Y, psidY, precon)
        H = regularize_h(H, lambda_min)
        if isinstance(H, np.ndarray) and H.dtype != dtype:
            H = H.astype(dtype, copy=False)

        direction = -solveh(G, H)
        if isinstance(direction, np.ndarray) and direction.dtype != dtype:
            direction = direction.astype(dtype, copy=False)

        success, new_Y, new_W, new_loss, _ = linesearch(
            Y, W, direction, current_loss, ls_tries
        )

        if not success:
            direction = (-G).astype(dtype, copy=False)
            _, new_Y, new_W, new_loss, _ = linesearch(
                Y, W, direction, current_loss, ls_tries
            )

        if isinstance(new_Y, np.ndarray) and new_Y.dtype != dtype:
            new_Y = new_Y.astype(dtype, copy=False)
        if isinstance(new_W, np.ndarray) and new_W.dtype != dtype:
            new_W = new_W.astype(dtype, copy=False)

        Y = new_Y
        W = new_W
        current_loss = new_loss

        if callback is not None:
            callback(locals())

        if verbose:
            info = "iteration %d, gradient norm = %.4g" % (n, gradient_norm)
            ending = "\r" if verbose == 1 else "\n"
            print(info, end=ending)

    return Y, W


if __name__ == "__main__":
    N, T = 10, 10000
    rng = np.random.RandomState(1)
    S = rng.laplace(size=(N, T)).astype(np.float32)
    A = rng.randn(N, N).astype(np.float32)
    X = A @ S
    simple_quasi_newton_ica(X, verbose=True)
