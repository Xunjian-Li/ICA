# """
# Python implementation of the Truncated Newton's method for ICA.
# Reference for the algorithm without preconditioning:
# Tillet, P. et al., "Infomax-ICA using Hessian-free optimization"
# """

# # Authors: Pierre Ablin <pierre.ablin@inria.fr>
# #          Alexandre Gramfort <alexandre.gramfort@inria.fr>
# #          Jean-Francois Cardoso <cardoso@iap.fr>
# #
# # License: BSD (3-clause)


# from __future__ import print_function
# from time import time
# from itertools import product
# import numpy as np
# import scipy.sparse as sparse
# from scipy import linalg
# import scipy.sparse.linalg as slinalg

# from ml_ica.tools import (loss, gradient, compute_h, regularize_h, solveh,
#                           hessian_free, score, score_der, linesearch)


# def truncated_ica(X, max_iter=100, tol=1e-7, l_fact=2., cg_tol=1e-2,
#                   cg_max=300, ls_tries=10, verbose=0, callback=None):
#     '''
#     The smallest eigenvalue of the Hessian is explicitly computed, but that
#     duration is not taken into account.

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

#     l_fact : float
#         Used to regularize the full Hessian. Its eigen values are shiffted by
#         l_fact * its smallest eigenvalue

#     cg_tol : float
#         Conjugate gradient stoping tolerance.

#     cg_max : float
#         Maximum number of inner conjugate gradient iterations

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
#     N, T = X.shape
#     Y = X.copy()
#     W = np.eye(N)
#     W_old = np.ones_like(W)
#     current_loss = loss(Y, W)
#     t0 = time()
#     timing = 0.
#     t_cheats = 0.
#     for n in range(max_iter):
#         # Compute the score and its derivative
#         psiY = score(Y)
#         psidY = score_der(psiY)
#         # Compute the gradient
#         G = gradient(Y, psiY)
#         # Stopping criterion
#         gradient_norm = linalg.norm(G.ravel(), ord=np.inf)
#         if callback is not None:
#             callback(locals())
#         diff_W = np.sum(np.abs(W - W_old)) / (N * N)
#         if n > 0 and diff_W < tol:
#             break
#         W_old = W.copy()
#         # Compute the smallest eigenvalue of H, freezing time.
#         t_h = time()
#         l_min = true_eigenvalue(Y, psidY)
#         t_cheat = time() - t_h
#         t_cheats += t_cheat
#         timing = time() - t0 - t_cheats
#         # Regularisation constant
#         l_reg = - l_fact * min(l_min, 0.)
#         # Compute the approximation
#         h = compute_h(Y, psidY)
#         # Regularize it
#         h = regularize_h(h, 1., 1)
#         # Compute the direction by conjugate gradient
#         direction = conjugate_gradient(Y, psidY, h, -G, l_reg, cg_max, cg_tol)
#         # Do a line search in that direction
#         success, Y_new, W_new, new_loss, _ =\
#             linesearch(Y, W, direction, current_loss, ls_tries)
#         # If it fails, fall back to gradient
#         if not success:
#             direction = - G
#             _, Y_new, W_new, new_loss, _ =\
#                 linesearch(Y, W, direction, current_loss, ls_tries)
#         # Update
#         Y = Y_new
#         W = W_new
#         current_loss = new_loss
#         if verbose:
#             info = 'iteration %d, gradient norm = %.4g' % (n, gradient_norm)
#             ending = '\r' if verbose == 1 else '\n'
#             print(info, end=ending)
#     return Y, W


# def full_hessian(Y, psidY):
#     '''
#     Computes the full hessian, in a sparse matrix. Very slow.
#     '''
#     N, T = Y.shape
#     # log det part
#     ind_ld = np.array(list(product(range(N), repeat=2)), dtype=int)
#     I, J = ind_ld.T
#     K = J
#     L = I
#     data = np.ones(N ** 2)
#     row = J + N * I
#     col = L + N * K
#     H_ld = sparse.coo_matrix((data, (row, col))).tocsr()
#     # density part
#     values = np.zeros((N, N, N))

#     for i in range(N):
#         temp = psidY[i, :]
#         values[i, :, :] = np.dot(temp[None, :] * Y, Y.T)
#     values /= float(T)
#     ind_sc = np.array(list(product(range(N), repeat=3)), dtype=int)
#     I, J, L = ind_sc.T
#     K = I
#     data = values[I, J, L]
#     row = J + N * I
#     col = L + N * K
#     H_d = sparse.coo_matrix((data, (row, col))).tocsr()
#     return H_d + H_ld


# def true_eigenvalue(Y, psidY):
#     '''
#     Computes the smallest eigenvalues of the true Hessian. Slow.
#     '''
#     H_full = full_hessian(Y, psidY)
#     return slinalg.eigsh(H_full, k=1, which='SA')[0][0]


# def conjugate_gradient(Y, psidY, h, G, lambda_reg, n_cg_it, tol):
#     '''
#     Uses the conjugate gradient method to compute the Newton direction H^-1 G.
#     We take advantage of the Hessian free product, and precondition the
#     algorithm with the hessian approximation h.
#     '''
#     x = np.zeros_like(G)
#     r = G.copy()
#     z = solveh(r, h)
#     p = z
#     rz = np.dot(r.ravel(), z.ravel())
#     for i in range(n_cg_it):
#         Ap = hessian_free(p, Y, psidY, lambda_reg)
#         pAp = np.dot(p.ravel(), Ap.ravel())
#         a = rz / pAp
#         x += a * p
#         r -= a * Ap
#         r_norm = np.sqrt(np.dot(r.ravel(), r.ravel()))
#         if r_norm / np.max(x) < tol:
#             break
#         z = solveh(r, h)
#         rz_old = rz
#         rz = np.dot(r.ravel(), z.ravel())
#         b = rz / rz_old
#         p = z + b * p
#     return x


# if __name__ == '__main__':
#     N, T = 10, 1000
#     rng = np.random.RandomState(1)
#     S = rng.laplace(size=(N, T))
#     A = rng.randn(N, N)
#     X = np.dot(A, S)
#     truncated_ica(X, verbose=True, max_iter=100)


# path: faster-ica/ml_ica/algorithms/truncated_newton.py

from __future__ import print_function
from time import time
from itertools import product

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as slinalg
from scipy import linalg

from ml_ica.tools import (
    loss, gradient, compute_h, regularize_h, solveh,
    hessian_free, score, score_der, linesearch
)


def truncated_ica(
    X,
    max_iter=100,
    tol=1e-7,
    l_fact=2.0,
    cg_tol=1e-2,
    cg_max=300,
    ls_tries=10,
    verbose=0,
    callback=None,
):
    """
    Truncated Newton ICA.

    Precision support
    -----------------
    - Preserves dtype of X for Y/W/G/h/direction (float32 stays float32).
    - Keeps the smallest-eigenvalue computation (sparse eigsh) in float64
      for stability and SciPy compatibility.
    """
    X = np.asarray(X, order="F")
    dtype = X.dtype
    if dtype not in (np.float32, np.float64):
        X = X.astype(np.float64, copy=False)
        dtype = X.dtype

    N, T = X.shape
    Y = X.copy(order="F")
    W = np.eye(N, dtype=dtype)
    W_old = np.ones_like(W)

    current_loss = loss(Y, W)
    t0 = time()
    t_cheats = 0.0

    for n in range(max_iter):
        # Compute score + derivative
        psiY = score(Y)
        psidY = score_der(psiY)

        # Gradient (keep dtype)
        G = gradient(Y, psiY).astype(dtype, copy=False)

        gradient_norm = linalg.norm(G.ravel(), ord=np.inf)

        if callback is not None:
            callback(locals())

        diff_W = np.sum(np.abs(W - W_old)) / (N * N)
        if n > 0 and diff_W < tol:
            break

        W_old = W.copy()

        # --- smallest eigenvalue of true Hessian (float64 path) ---
        t_h = time()
        l_min = true_eigenvalue(Y.astype(np.float64, copy=False),
                               psidY.astype(np.float64, copy=False))
        t_cheats += (time() - t_h)

        timing = time() - t0 - t_cheats

        # Regularization constant (scalar)
        l_reg = -float(l_fact) * min(float(l_min), 0.0)

        # Hessian approximation + regularize (cast back if needed)
        h = compute_h(Y, psidY)
        h = regularize_h(h, 1.0, 1)
        if isinstance(h, np.ndarray) and h.dtype != dtype:
            h = h.astype(dtype, copy=False)

        # Newton direction via CG (dtype)
        direction = conjugate_gradient(Y, psidY, h, (-G).astype(dtype, copy=False),
                                       l_reg, cg_max, cg_tol)

        # Line search
        success, Y_new, W_new, new_loss, _ = linesearch(
            Y, W, direction, current_loss, ls_tries
        )
        if not success:
            direction = (-G).astype(dtype, copy=False)
            _, Y_new, W_new, new_loss, _ = linesearch(
                Y, W, direction, current_loss, ls_tries
            )

        if isinstance(Y_new, np.ndarray) and Y_new.dtype != dtype:
            Y_new = Y_new.astype(dtype, copy=False)
        if isinstance(W_new, np.ndarray) and W_new.dtype != dtype:
            W_new = W_new.astype(dtype, copy=False)

        Y = Y_new
        W = W_new
        current_loss = new_loss

        if verbose:
            info = "iteration %d, gradient norm = %.4g" % (n, gradient_norm)
            ending = "\r" if verbose == 1 else "\n"
            print(info, end=ending)

    return Y, W


def full_hessian(Y, psidY):
    """
    Computes the full Hessian as a sparse matrix (very slow).
    This is used only for the eigenvalue cheat step; float64 is fine.
    """
    N, T = Y.shape

    ind_ld = np.array(list(product(range(N), repeat=2)), dtype=int)
    I, J = ind_ld.T
    K = J
    L = I
    data = np.ones(N ** 2)
    row = J + N * I
    col = L + N * K
    H_ld = sparse.coo_matrix((data, (row, col))).tocsr()

    values = np.zeros((N, N, N))
    for i in range(N):
        temp = psidY[i, :]
        values[i, :, :] = np.dot(temp[None, :] * Y, Y.T)
    values /= float(T)

    ind_sc = np.array(list(product(range(N), repeat=3)), dtype=int)
    I, J, L = ind_sc.T
    K = I
    data = values[I, J, L]
    row = J + N * I
    col = L + N * K
    H_d = sparse.coo_matrix((data, (row, col))).tocsr()

    return H_d + H_ld


def true_eigenvalue(Y, psidY):
    """
    Computes the smallest eigenvalue of the true Hessian (slow).
    """
    H_full = full_hessian(Y, psidY)
    return slinalg.eigsh(H_full, k=1, which="SA")[0][0]


def conjugate_gradient(Y, psidY, h, G, lambda_reg, n_cg_it, tol):
    """
    Conjugate gradient to solve (H + lambda_reg I) x = G with Hessian-free products.
    Keeps dtype consistent with G/Y.
    """
    dtype = G.dtype
    x = np.zeros_like(G)
    r = G.copy()
    z = solveh(r, h)
    if isinstance(z, np.ndarray) and z.dtype != dtype:
        z = z.astype(dtype, copy=False)

    p = z
    rz = np.dot(r.ravel(), z.ravel())

    eps = np.finfo(dtype).eps if dtype in (np.float32, np.float64) else 1e-12

    for _ in range(n_cg_it):
        Ap = hessian_free(p, Y, psidY, lambda_reg)
        if isinstance(Ap, np.ndarray) and Ap.dtype != dtype:
            Ap = Ap.astype(dtype, copy=False)

        pAp = np.dot(p.ravel(), Ap.ravel())
        if pAp == 0:
            break

        a = rz / pAp
        x += a * p
        r -= a * Ap

        r_norm = np.sqrt(np.dot(r.ravel(), r.ravel()))
        denom = max(float(np.max(np.abs(x))), eps)
        if (r_norm / denom) < tol:
            break

        z = solveh(r, h)
        if isinstance(z, np.ndarray) and z.dtype != dtype:
            z = z.astype(dtype, copy=False)

        rz_old = rz
        rz = np.dot(r.ravel(), z.ravel())
        if rz_old == 0:
            break

        b = rz / rz_old
        p = z + b * p

    return x
