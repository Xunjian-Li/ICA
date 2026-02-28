# import numpy as np
# from .derivatives import loss


# def linesearch(Y, W, direction, initial_loss=None, n_ls_tries=10):
#     '''
#     Performs a simple backtracking linesearch in the direction "direction".
#     Does n_ls_tries attempts before exiting.
#     '''
#     N = Y.shape[0]
#     W_proj = np.dot(direction, W)
#     step = 1.
#     if initial_loss is None:
#         initial_loss = loss(Y, W)
#     for n in range(n_ls_tries):
#         new_Y = np.dot(np.eye(N) + step * direction, Y)
#         new_W = W + step * W_proj
#         new_loss = loss(new_Y, new_W)
#         if new_loss < initial_loss:
#             success = True
#             break
#         step /= 2.
#     else:
#         success = False
#     return success, new_Y, new_W, new_loss, step

import numpy as np
from .derivatives import loss


def linesearch(Y, W, direction, initial_loss=None, n_ls_tries=10):
    """
    Backtracking line search (dtype-preserving).

    Keeps computations in Y.dtype (float32 stays float32) by:
      - using dtype-aware identity and scalars
      - ensuring intermediate arrays stay in the same dtype
    """
    Y = np.asarray(Y, order="F")
    W = np.asarray(W)
    direction = np.asarray(direction)

    dtype = Y.dtype
    if dtype not in (np.float32, np.float64):
        dtype = np.float64
        Y = Y.astype(dtype, copy=False)
        W = W.astype(dtype, copy=False)
        direction = direction.astype(dtype, copy=False)

    N = Y.shape[0]
    I = np.eye(N, dtype=dtype)

    W_proj = (direction @ W).astype(dtype, copy=False)

    step = dtype.type(1.0)

    if initial_loss is None:
        initial_loss = loss(Y, W)

    success = False
    new_Y = Y
    new_W = W
    new_loss = initial_loss

    for _ in range(n_ls_tries):
        A = (I + step * direction).astype(dtype, copy=False)
        new_Y = (A @ Y).astype(dtype, copy=False)
        new_W = (W + step * W_proj).astype(dtype, copy=False)

        new_loss = loss(new_Y, new_W)
        if new_loss < initial_loss:
            success = True
            break

        step = step / dtype.type(2.0)

    return success, new_Y, new_W, new_loss, float(step)
