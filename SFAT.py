import warnings
from collections import namedtuple
from typing import Literal, Optional, Sequence, TypedDict

import numpy as np
import numpy.typing as npt
from scipy.linalg import issymmetric, lstsq, solve, svd
from scipy.stats import multivariate_normal

SMALL = 1e-12
_LOG_2_PI = np.log(2 * np.pi)
RANDOM_VARIANCE_FLAG = -1

MVNParam = namedtuple("MVNParam", ["mean", "cov"])


class InitDict(TypedDict, total=False):
    W_s: np.ndarray
    W_t: np.ndarray
    W_y: np.ndarray
    B_s: np.ndarray
    B_t: np.ndarray
    var_s: np.ndarray
    var_t: np.ndarray
    var_y: np.ndarray


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


def add_diag(M, value):
    """Add value to the diagnal of a matrix inplace, and return self"""
    M.flat[: M.shape[1] ** 2 : M.shape[1] + 1] += value
    return M


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


def my_svd(X, n_components):
    _, s, Vh = svd(X, full_matrices=False)
    return (
        s[:n_components],
        Vh[:n_components],
        np.square(s[n_components:]).sum(),
    )


def _diag_dot(A, B, C):
    """return diag(A @ B @ C)"""
    return np.einsum("ij,jk,ki->i", A, B, C)


def _householder_rotate_bases(A: npt.NDArray[np.float64]):
    """Use householder transform to find rotation matrix that rotate the bases of `A` (row space) to orthonormal bases"""
    *_, Vh = np.linalg.svd(A, full_matrices=False)
    R = np.eye(Vh.shape[1])
    for i in range(Vh.shape[0]):
        v = Vh[i, :]
        e = np.zeros_like(v)
        e[i] = 1
        u = v - e
        u = u / np.linalg.norm(u)
        P = np.eye(u.size) - 2 * np.outer(u, u)
        Vh = Vh @ P
        R = R @ P
    return R


def _rotate_bottom_to_unit(A):
    """Use householder transform to find a rotation that make the bottom row vector be a unit vector."""
    v = A[-1, :].copy()
    v[0] -= -np.sign(v[0]) * np.linalg.norm(v)
    v /= np.linalg.norm(v)
    Q = add_diag(-2 * np.outer(v, v), 1)
    return Q


def _householder_rotate_triangle(A):
    """Use householder transform to let the matrix become left up triangle"""
    n_row, n_col = A.shape
    A_ = A.copy()
    Q = np.eye(n_col)
    for i in range(n_row):
        R_ = _rotate_bottom_to_unit(A_[: n_row - i, i:])
        A_[: n_row - i, i:] = A_[: n_row - i, i:] @ R_
        Q[:, i:] = Q[:, i:] @ R_
    return Q


def _householder_rotate(A, type: Literal["triangle", "bases"] = "bases"):
    if type == "bases":
        return _householder_rotate_bases(A)
    elif type == "triangle":
        return _householder_rotate_triangle(A)
    else:
        raise ValueError()


def _init_svd(X, n_comp):
    _, _, Ah = svd(X)
    A = Ah.T[:, :n_comp]
    if A.shape[1] != n_comp:
        raise NotImplementedError()
    return A


def _meaningful_rotate(A, method="varimax", tol=1e-6, max_iter=100):
    """Return rotation_matrix."""
    nrow, ncol = A.shape
    rotation_matrix = np.eye(ncol)
    var = 0

    for _ in range(max_iter):
        comp_rot = np.dot(A, rotation_matrix)
        if method == "varimax":
            tmp = comp_rot * np.transpose((comp_rot**2).sum(axis=0) / nrow)
        elif method == "quartimax":
            tmp = 0
        u, s, v = np.linalg.svd(np.dot(A.T, comp_rot**3 - tmp))
        rotation_matrix = np.dot(u, v)
        var_new = np.sum(s)
        if var != 0 and var_new < var * (1 + tol):
            break
        var = var_new

    return rotation_matrix


def _return_as(
    data: list[MVNParam], to: Literal["array", "dist", "namedtuple"] = "dist"
):
    if to == "array":
        mean, cov = zip(*data)
        return np.stack(mean, axis=0), np.stack(cov, axis=0)
    elif to == "dist":
        return [multivariate_normal(mean=mvn.mean, cov=mvn.cov) for mvn in data]
    elif to == "namedtuple":
        return data


class SpervisedLVGA:
    def __init__(
        self,
        n_share_comp,
        n_spec_comp,
        variance_constraints: (
            Sequence[int] | Literal["equal", "domain", "none"]
        ) = "none",
        solver: Literal["svd", "em"] = "svd",
        init_method: Literal["svd", "random", "custom"] = "svd",
        init_values: InitDict | None = None,
        rotation: Literal["varimax", "quartimax"] | None = None,
        max_iter=5000,
        tol=1e-4,
    ) -> None:
        self.n_share_comp = n_share_comp
        self.n_spec_comp = n_spec_comp
        self.variance_constraints = variance_constraints
        self.solver = solver
        self.init_method = init_method
        self.init_values = init_values
        self.rotation = rotation
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, Xs, Xt, Y):
        Xs, Xt, Y = np.atleast_2d(Xs, Xt, Y)

        Xs, self.Xs_mean_ = center(Xs)
        Xt, self.Xt_mean_ = center(Xt)
        Y, self.Y_mean_ = center(Y)

        if self.solver == "em":
            (W_s, W_t, W_y), (B_s, B_t), (var_s, var_t, var_y) = self._fit_em(Xs, Xt, Y)
        elif self.solver == "svd":
            (W_s, W_t, W_y), (B_s, B_t), (var_s, var_t, var_y) = self._fit_svd(
                Xs, Xt, Y, self.max_iter
            )

        self.W_s = W_s
        self.W_t = W_t
        self.W_y = W_y
        self.B_s = B_s
        self.B_t = B_t

        self.var_s = var_s
        self.var_t = var_t
        self.var_y = var_y

        return self

    def _fit_em(self, Xs, Xt, Y):
        """`Xs`, `Xt`, and `Y` need to be centered."""

        n_samples = Xs.shape[0]
        nfeat_s = Xs.shape[1]
        nfeat_t = Xt.shape[1]
        nfeat_y = Y.shape[1]
        total_comp = self.n_share_comp + self.n_spec_comp
        V = np.hstack((Xs, Xt, Y))

        if self.init_method == "svd":
            A_s = _init_svd(Xs, total_comp)
            A_t = _init_svd(Xt, total_comp)
            W_y = _init_svd(Y, self.n_share_comp)
        elif self.init_method == "random":
            A_s = np.random.rand(nfeat_s, total_comp)
            A_t = np.random.rand(nfeat_t, total_comp)
            W_y = np.random.rand(nfeat_y, self.n_share_comp)
        else:
            W_s = self.init_values["W_s"]
            W_t = self.init_values["W_t"]
            W_y = self.init_values["W_y"]
            B_s = self.init_values["B_s"]
            B_t = self.init_values["B_t"]
            A_s = np.hstack((W_s, B_s))
            A_t = np.hstack((W_t, B_t))

        Phi = np.ones(nfeat_s + nfeat_t + nfeat_y)
        if self.init_method == "custom":
            Phi[:nfeat_s] = self.init_values["var_s"]
            Phi[nfeat_s:-nfeat_y] = self.init_values["var_t"]
            Phi[-nfeat_y:] = self.init_values["var_y"]

        if not isinstance(self.variance_constraints, str):
            var_lbls = np.asarray(self.variance_constraints).squeeze()
            if var_lbls.shape != Phi.shape:
                raise ValueError(
                    "The shape of variance_constraints mismatch the number of variables."
                )
            ulbls = np.unique(var_lbls)
            ulbls = ulbls[ulbls != RANDOM_VARIANCE_FLAG]

        loglike = []
        llconst = (nfeat_s + nfeat_t + nfeat_y) * _LOG_2_PI
        S = np.cov(V, rowvar=False, ddof=0)
        old_ll = -np.inf
        A = np.zeros((nfeat_s + nfeat_t + nfeat_y, total_comp), dtype=float)
        for it in range(self.max_iter):
            A[:nfeat_s, :] = A_s
            A[nfeat_s : nfeat_s + nfeat_t, :] = A_t
            A[-nfeat_y:, : self.n_share_comp] = W_y

            Cov_Z = np.linalg.inv(add_diag((A.T / Phi) @ A, 1))

            EZ = (V / Phi) @ A @ Cov_Z
            EZZ = n_samples * Cov_Z + EZ.T @ EZ

            # Log likelihood
            Sigma_v = add_diag(A @ A.T, Phi)
            _, logdet = np.linalg.slogdet(Sigma_v)
            ll = llconst
            ll += logdet
            ll += np.trace(
                mldivide(Sigma_v, S, sym_a=True)
            )  # np.trace(np.linalg.inv(Sigma_v) @ S)
            ll *= -n_samples / 2
            loglike.append(ll)
            if (ll - old_ll) < self.tol:
                break
            old_ll = ll

            # Maximize
            t = mrdivide(EZ, EZZ)
            A_s = Xs.T @ t
            A_t = Xt.T @ t
            W_y = mrdivide(
                Y.T @ EZ[:, : self.n_share_comp],
                EZZ[: self.n_share_comp, : self.n_share_comp],
            )

            Phi = (
                np.sum(V**2, axis=0)
                - 2 * _diag_dot(A, EZ.T, V)
                + _diag_dot(A, EZZ, A.T)
            ) / n_samples

            if self.variance_constraints == "none":
                pass
            elif self.variance_constraints == "domain":
                Phi[:nfeat_s] = np.mean(Phi[:nfeat_s])
                Phi[nfeat_s:-nfeat_y] = np.mean(Phi[nfeat_s:-nfeat_y])
                Phi[-nfeat_y:] = np.mean(Phi[-nfeat_y:])
            elif self.variance_constraints == "equal":
                Phi[:] = Phi.mean()
            else:
                for lbl in ulbls:
                    Phi[var_lbls == lbl] = Phi[var_lbls == lbl].mean()
        else:
            warnings.warn("Max iteration was reached.")

        W_s, B_s = np.hsplit(A_s, [self.n_share_comp])
        W_t, B_t = np.hsplit(A_t, [self.n_share_comp])

        if self.rotation is not None:
            R_share = _householder_rotate(W_y)
            R_spec = _meaningful_rotate(np.vstack((B_s, B_t)), method=self.rotation)
            W_s = W_s @ R_share
            W_t = W_t @ R_share
            W_y = W_y @ R_share

            B_s = B_s @ R_spec
            B_t = B_t @ R_spec

        self.loglike_ = loglike
        var_s, var_t, var_y = np.split(Phi, [nfeat_s, nfeat_s + nfeat_t])
        return (W_s, W_t, W_y), (B_s, B_t), (var_s, var_t, var_y)

    def _fit_svd(self, Xs, Xt, Y, max_iter):
        """`Xs`, `Xt`, and `Y` need to be centered."""

        n_samples = Xs.shape[0]
        nfeat_s = Xs.shape[1]
        nfeat_t = Xt.shape[1]
        nfeat_y = Y.shape[1]
        V = np.hstack((Xs, Xt, Y))

        Phi = np.ones(nfeat_s + nfeat_t + nfeat_y)
        if self.init_method == "custom":
            Phi[:nfeat_s] = self.init_values["var_s"]
            Phi[nfeat_s:-nfeat_y] = self.init_values["var_t"]
            Phi[-nfeat_y:] = self.init_values["var_y"]

        if not isinstance(self.variance_constraints, str):
            var_lbls = np.asarray(self.variance_constraints).squeeze()
            if var_lbls.shape != Phi.shape:
                raise ValueError(
                    "The shape of variance_constraints mismatch the number of variables."
                )
            ulbls = np.unique(var_lbls)
            ulbls = ulbls[ulbls != RANDOM_VARIANCE_FLAG]

        loglike = []
        llconst = Phi.size * _LOG_2_PI + self.n_share_comp + self.n_spec_comp
        var_v = np.var(V, axis=0)  # Diag(S) = var(v)

        old_ll = -np.inf
        for it in range(max_iter):
            sqrt_psi = np.sqrt(Phi) + SMALL
            X_hat = V / (sqrt_psi * np.sqrt(n_samples))
            s, Uh, unexp_var = my_svd(
                X_hat, n_components=self.n_share_comp + self.n_spec_comp
            )
            s **= 2
            A = np.sqrt(np.maximum(s - 1.0, 0.0)) * Uh.T
            A *= sqrt_psi[:, np.newaxis]

            ll = llconst + np.sum(np.log(s))
            ll += unexp_var + np.sum(np.log(Phi))
            ll *= -n_samples / 2
            loglike.append(ll)
            if (ll - old_ll) < self.tol:
                break
            old_ll = ll

            Phi = np.maximum(SMALL, var_v - np.sum(A**2, axis=1))

            if self.variance_constraints == "none":
                pass
            elif self.variance_constraints == "domain":
                Phi[:nfeat_s] = np.mean(Phi[:nfeat_s])
                Phi[nfeat_s:-nfeat_y] = np.mean(Phi[nfeat_s:-nfeat_y])
                Phi[-nfeat_y:] = np.mean(Phi[-nfeat_y:])
            elif self.variance_constraints == "equal":
                Phi[:] = Phi.mean()
            else:
                for lbl in ulbls:
                    Phi[var_lbls == lbl] = Phi[var_lbls == lbl].mean()
        else:
            warnings.warn("Max iteration was reached.")

        As, At, Ay = np.vsplit(A, [nfeat_s, nfeat_s + nfeat_t])

        R = _householder_rotate(Ay)
        As = As @ R
        At = At @ R
        Ay = Ay @ R

        W_s, B_s = np.hsplit(As, [self.n_share_comp])
        W_t, B_t = np.hsplit(At, [self.n_share_comp])
        W_y, _ = np.hsplit(Ay, [self.n_share_comp])

        if self.rotation is not None:
            R_spec = _meaningful_rotate(np.vstack((B_s, B_t)), method=self.rotation)
            B_s = B_s @ R_spec
            B_t = B_t @ R_spec

        var_s, var_t, var_y = np.split(Phi, [nfeat_s, nfeat_s + nfeat_t])
        self.loglike_ = loglike
        return (W_s, W_t, W_y), (B_s, B_t), (var_s, var_t, var_y)

    def posterior(
        self,
        data,
        cond_on: Literal["Xs", "Xt", "Y", "Z"] = "Xt",
        pred_to: Literal["Xs", "Xt", "Y", "Z"] = "Xs",
        return_as: Literal["array", "dist", "namedtuple"] = "array",
    ):
        if cond_on == pred_to:
            raise ValueError()

        if cond_on == "Z":
            W = self.get_W(pred_to)
            B = self.get_B(pred_to)
            mu = self.get_mean(pred_to)
            var = self.get_var(pred_to)
            cov = np.diag(var)
            if pred_to == "Y":
                data = data[:, : self.n_share_comp]
                results = [MVNParam(mean=mu + W @ z, cov=cov) for z in data]
                return _return_as(results, to=return_as)
            A = np.vstack((W, B))
            results = [MVNParam(mean=mu + A @ z, cov=cov) for z in data]
            return _return_as(results, to=return_as)
        elif pred_to == "Z":
            W = self.get_W(cond_on)
            B = self.get_B(cond_on)
            mu = self.get_mean(cond_on)
            A = np.vstack((W, B))
            var = self.get_var(cond_on)

            inv_cov = add_diag((A.T / var) @ A, 1)
            cov = np.linalg.pinv(inv_cov, hermitian=True)
            cache_M = inv_cov @ (A.T / var)
            results = [
                MVNParam(mean=cache_M @ (x - mu), cov=cov, allow_singular=True)
                for x in data
            ]
            return _return_as(results, to=return_as)
        else:
            mu_1 = self.get_mean(pred_to)
            mu_2 = self.get_mean(cond_on)
            W_1 = self.get_W(pred_to)
            W_2 = self.get_W(cond_on)
            B_1 = self.get_B(pred_to)
            B_2 = self.get_B(cond_on)
            var_1 = self.get_var(pred_to)
            var_2 = self.get_var(cond_on)

            Sigma_11 = W_1 @ W_1.T + B_1 @ B_1.T
            Sigma_12 = W_1 @ W_2.T + B_1 @ B_2.T
            Sigma_21 = Sigma_12.T
            Sigma_22 = W_2 @ W_2.T + B_2 @ B_2.T

            Sigma_11 = add_diag(Sigma_11, var_1)
            Sigma_22 = add_diag(Sigma_22, var_2)

            cache_M = mrdivide(Sigma_12, Sigma_22)
            cov = Sigma_11 - cache_M @ Sigma_21

            results = [
                MVNParam(
                    mean=mu_1 + cache_M @ (x - mu_2),
                    cov=cov,
                )
                for x in data
            ]
            return _return_as(results, to=return_as)

    def get_mean(self, varname: Literal["Xs", "Xt", "Y", "Z"]):
        match varname:
            case "Xs":
                return self.Xs_mean_
            case "Xt":
                return self.Xt_mean_
            case "Y":
                return self.Y_mean_
            case _:
                raise ValueError()

    def get_W(self, varname: Literal["Xs", "Xt", "Y"]):
        match varname:
            case "Xs":
                return self.W_s
            case "Xt":
                return self.W_t
            case "Y":
                return self.W_y
            case _:
                raise ValueError()

    def get_B(self, varname: Literal["Xs", "Xt", "Y"]):
        match varname:
            case "Xs":
                return self.B_s
            case "Xt":
                return self.B_t
            case "Y":
                return np.zeros((self.W_y.shape[0], self.B_t.shape[1]))
            case _:
                raise ValueError()

    def get_var(self, varname: Literal["Xs", "Xt", "Y"]):
        match varname:
            case "Xs":
                return self.var_s
            case "Xt":
                return self.var_t
            case "Y":
                return self.var_y
            case _:
                raise ValueError()

    def loglikelihood(
        self, Xs, Xt, Y, reduction: Literal["mean", "sum", "raw"] = "mean"
    ):
        Xs = Xs - self.Xs_mean_
        Xt = Xt - self.Xt_mean_
        Y = Y - self.Y_mean_

        full_data = np.hstack((Xs, Xt, Y))
        A = np.block(
            [
                [self.W_s, self.B_s],  # A_s
                [self.W_t, self.B_t],  # A_t
                [self.W_y, np.zeros((self.W_y.shape[0], self.B_t.shape[1]))],  # A_y
            ]
        )
        Phi = np.ones(A.shape[0])
        nfeat_s = self.W_s.shape[0]
        nfeat_y = self.W_y.shape[0]
        Phi[:nfeat_s] = self.var_s
        Phi[nfeat_s:-nfeat_y] = self.var_t
        Phi[-nfeat_y:] = self.var_y
        Sigma = add_diag(A @ A.T, Phi)
        # Sigma = A @ A.T
        # Sigma.flat[:: Sigma.shape[0] + 1] += Phi

        const = -(A.shape[0] * _LOG_2_PI + np.linalg.slogdet(Sigma)[1]) / 2

        ll = np.sum(mrdivide(full_data, Sigma) * full_data, axis=1) / 2

        if reduction == "raw":
            return ll + const
        elif reduction == "mean":
            return ll.mean() + const
        elif reduction == "sum":
            return ll.sum() + full_data.shape[0] * const
        else:
            raise ValueError()

    def transform(
        self,
        data,
        cond_on: Literal["Xs", "Xt", "Y", "Z"] = "Xt",
        pred_to: Literal["Xs", "Xt", "Y", "Z"] = "Xs",
    ):
        return self.posterior(
            data, cond_on=cond_on, pred_to=pred_to, return_as="array"
        )[0]
