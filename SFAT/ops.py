from typing import Optional
from scipy.linalg import issymmetric, solve, lstsq
import numpy as np


def mldivide(A, B, sym_a: Optional[bool] = None):
    r"""same as A\B (in matlab) or pinv(A) @ B"""
    if A.ndim == 1:
        raise ValueError("The first argument needs to be a matrix.")
    if A.ndim == 2 and A.shape[0] == A.shape[1]:
        if sym_a == True or (sym_a is None and issymmetric(A)):
            return solve(A, B, assume_a="sym")
        else:
            return solve(A, B)
    else:
        return lstsq(A, B)[0]


def mrdivide(A, B, sym_b: Optional[bool] = None):
    """same as A/B (in matlab) or A @ pinv(B)"""
    if B.ndim == 1:
        raise ValueError("The second argument needs to be a matrix.")
    if A.ndim == 1:
        A = A[None, :]
        return mldivide(B.T, A.T, sym_a=sym_b).squeeze(1)
    return mldivide(B.T, A.T, sym_a=sym_b).T


def add_diag(M, value):
    """Add value to the diagnal of a matrix inplace, and return self"""
    M.flat[: M.shape[1] ** 2 : M.shape[1] + 1] += value
    return M


def center(X, axis=0):
    """center data

    Parameters
    ----------
    X : ndarray
        Data matrix
    axis : int, optional
        centered axis, by default 0

    Returns
    -------
    X_hat : ndarray
        Centered X
    mean : ndarray
        Means of the original X
    """
    mean = np.mean(X, axis=axis, keepdims=True)
    return X - mean, np.squeeze(mean, axis=axis)
