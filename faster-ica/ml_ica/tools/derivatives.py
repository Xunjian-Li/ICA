# import numpy as np
# import numexpr as ne


# def score(Y):
#     '''
#     Returns the score function evaluated for each sample
#     '''
#     return ne.evaluate('tanh(Y / 2)')


# def score_der(psiY):
#     '''
#     Returns the derivative of the score
#     '''
#     return ne.evaluate('(- psiY ** 2 + 1.) / 2.')


# def loss(Y, W):
#     '''
#     Computes the loss function for (Y, W)
#     '''
#     T = Y.shape[1]
#     log_det = np.linalg.slogdet(W)[1]
#     logcoshY = np.sum(ne.evaluate('abs(Y) + 2. * log1p(exp(-abs(Y)))'))
#     return - log_det + logcoshY / float(T)


# def gradient(Y, psiY):
#     '''
#     Returns the gradient at Y, using the score psiY
#     '''
#     N, T = Y.shape
#     return np.inner(psiY, Y) / float(T) - np.eye(N)


# def compute_h(Y, psidY, precon=2):
#     '''
#     Returns the diagonal coefficients of H 1/ H2 in a N x N matrix
#     '''
#     N, T = Y.shape
#     if precon == 2:
#         return np.inner(psidY, Y ** 2) / float(T)
#     else:
#         Y_squared = Y ** 2
#         sigma2 = np.mean(Y_squared, axis=1)
#         psidY_mean = np.mean(psidY, axis=1)
#         h1 = psidY_mean[:, None] * sigma2[None, :]
#         diagonal_term = np.mean(Y_squared * psidY)
#         h1[np.diag_indices_from(h1)] = diagonal_term
#         return h1


# def regularize_h(h, lambda_min, mode=0):
#     '''
#     Regularizes the hessian approximation h using the constant lambda_min.
#     Mode selects the regularization algorithm
#     0 -> Shift each eigenvalue below lambda_min to lambda_min
#     1 -> add lambda_min x Id to h
#     '''
#     if mode == 0:
#         # Compute the eigenvalues of the Hessian
#         eigenvalues = 0.5 * (h + h.T - np.sqrt((h-h.T) ** 2 + 4.))
#         # Regularize
#         problematic_locs = eigenvalues < lambda_min
#         np.fill_diagonal(problematic_locs, False)
#         i_pb, j_pb = np.where(problematic_locs)
#         h[i_pb, j_pb] += lambda_min - eigenvalues[i_pb, j_pb]
#     if mode == 1:
#         h += lambda_min
#     return h


# def solveh(G, h):
#     '''
#     Returns H^-1 G
#     '''
#     return (G * h.T - G.T) / (h * h.T - 1.)


# def hessian_free(M, Y, psidY, l_reg=0.):
#     '''
#     Computes the Hessian free product (H + l_reg * Id)M where H is the true
#     Hessian, for a N x N matrix M.
#     '''
#     T = Y.shape[1]
#     return l_reg * M + M.T + np.inner(psidY * np.dot(M, Y), Y) / float(T)


# path: faster-ica/ml_ica/tools.py   (or wherever these live)

import numpy as np
import numexpr as ne


def _dtype_like(*arrs):
    for a in arrs:
        if isinstance(a, np.ndarray):
            if a.dtype in (np.float32, np.float64):
                return a.dtype
            return a.dtype
    return np.float64


def score(Y):
    """Returns the score function evaluated for each sample."""
    Y = np.asarray(Y)
    # numexpr preserves dtype for float32/float64 inputs
    return ne.evaluate("tanh(Y / 2)")


def score_der(psiY):
    """Returns the derivative of the score."""
    psiY = np.asarray(psiY)
    dtype = psiY.dtype
    one = np.array(1.0, dtype=dtype)
    two = np.array(2.0, dtype=dtype)
    return ne.evaluate("(-psiY**2 + one) / two")


def loss(Y, W):
    """Computes the loss function for (Y, W)."""
    Y = np.asarray(Y)
    W = np.asarray(W)
    dtype = _dtype_like(Y)

    T = Y.shape[1]
    log_det = np.linalg.slogdet(W)[1]  # returns float64; fine
    # logcosh(Y) computed in dtype of Y by numexpr; sum promotes to float64 anyway
    logcoshY = np.sum(ne.evaluate("abs(Y) + 2. * log1p(exp(-abs(Y)))"))
    return -log_det + logcoshY / float(T)


def gradient(Y, psiY):
    """Returns the relative gradient at Y using score psiY."""
    Y = np.asarray(Y)
    psiY = np.asarray(psiY)
    dtype = Y.dtype
    N, T = Y.shape

    out = np.inner(psiY, Y) / dtype.type(T)
    out = out.astype(dtype, copy=False)
    out -= np.eye(N, dtype=dtype)
    return out


def compute_h(Y, psidY, precon=2):
    """Returns diagonal coefficients of H1/H2 in an N×N matrix."""
    Y = np.asarray(Y)
    psidY = np.asarray(psidY)
    dtype = Y.dtype
    N, T = Y.shape
    invT = dtype.type(1.0) / dtype.type(T)

    if precon == 2:
        out = np.inner(psidY, Y ** 2) * invT
        return out.astype(dtype, copy=False)

    Y_squared = Y ** 2
    sigma2 = np.mean(Y_squared, axis=1).astype(dtype, copy=False)
    psidY_mean = np.mean(psidY, axis=1).astype(dtype, copy=False)
    h1 = psidY_mean[:, None] * sigma2[None, :]
    diagonal_term = np.mean(Y_squared * psidY).astype(dtype, copy=False)
    h1 = h1.astype(dtype, copy=False)
    h1[np.diag_indices_from(h1)] = diagonal_term
    return h1


def regularize_h(h, lambda_min, mode=0):
    """Regularizes Hessian approximation h."""
    h = np.asarray(h)
    dtype = h.dtype
    lam = dtype.type(lambda_min)

    if mode == 0:
        # Compute the eigenvalues of the Hessian-like operator (dtype-safe)
        # NOTE: sqrt may upcast internally; cast back.
        disc = (h - h.T) ** 2 + dtype.type(4.0)
        disc = np.maximum(disc, dtype.type(0.0))
        eigenvalues = dtype.type(0.5) * (h + h.T - np.sqrt(disc).astype(dtype, copy=False))

        problematic = eigenvalues < lam
        np.fill_diagonal(problematic, False)
        i_pb, j_pb = np.where(problematic)
        if i_pb.size:
            h[i_pb, j_pb] += lam - eigenvalues[i_pb, j_pb]

    elif mode == 1:
        h += lam

    return h


def solveh(G, h):
    """Returns H^{-1} G for diagonal Hessian approximation h."""
    G = np.asarray(G)
    h = np.asarray(h)
    dtype = G.dtype
    denom = h * h.T - dtype.type(1.0)
    return ((G * h.T - G.T) / denom).astype(dtype, copy=False)


def hessian_free(M, Y, psidY, l_reg=0.0):
    """
    Computes Hessian-free product (H + l_reg*I)M where H is the true Hessian.
    """
    M = np.asarray(M)
    Y = np.asarray(Y)
    psidY = np.asarray(psidY)
    dtype = M.dtype
    T = Y.shape[1]

    lreg = dtype.type(l_reg)
    invT = dtype.type(1.0) / dtype.type(T)

    MY = np.dot(M, Y).astype(dtype, copy=False)
    inner_term = np.inner(psidY * MY, Y) * invT
    return (lreg * M + M.T + inner_term).astype(dtype, copy=False)
