"""
Non-amortized kernel-GLM posterior parameterization for latent SDE models
Implementation follows SVISE (Course and Nair, 2023)
See https://github.com/coursekevin/svise/blob/main/svise
"""

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flax import linen as nn

from typing import Any, Callable, Dict, Optional, Tuple

from helmholtz_sde.posterior.posterior import Posterior
from helmholtz_sde.utils.general_helpers import inverse_softplus


def kumaraswamy_warp(t: jnp.array, t0: float, tf: float, raw_alpha: jnp.array, raw_beta: jnp.array) -> jnp.array:
    """
    Monotonic time warp via the Kumaraswamy CDF: w(u) = 1 - (1 - u^alpha)^beta
    Maps t in [t0, tf] to warped time in [t0, tf]
    """
    alpha = jax.nn.softplus(raw_alpha)
    beta = jax.nn.softplus(raw_beta)
    dt = tf - t0
    u = jnp.clip((t - t0) / dt, 1e-10, 1.0 - 1e-10)
    diff = jnp.maximum(1.0 - u ** alpha, 1e-10)
    return (1.0 - diff ** beta) * dt + t0


def identity_warp(t: jnp.array, t0: float, tf: float, raw_alpha: jnp.array, raw_beta: jnp.array) -> jnp.array:
    """
    Returns t unchanged
    """
    return t


def matern52_kernel(x1: jnp.array, x2: jnp.array, raw_len: jnp.array, raw_sigf: jnp.array) -> jnp.array:
    """
    Matern 5/2 kernel on pre-warped 1D inputs
    """
    l = jax.nn.softplus(raw_len)
    sig2 = jax.nn.softplus(raw_sigf) ** 2
    # Mean-center for numerical stability
    mean = jnp.mean(x1, axis=0)
    diff = ((x1 - mean)[:, None, :] - (x2 - mean)[None, :, :])[..., 0]
    r = jnp.sqrt(diff ** 2 + 1e-20) * math.sqrt(5) / l
    K = jnp.exp(-r) * (r ** 2 / 3 + r + 1) * sig2
    return K


def _solve_ls_svd(A: jnp.array, y: jnp.array, gamma: float = 1e-6) -> jnp.array:
    """
    Regularized least squares via truncated SVD
    """
    U, S, Vt = jnp.linalg.svd(A, full_matrices=False)
    return Vt.T @ ((U.T @ y) * (S / (S ** 2 + gamma))[:, None])


def _cross_validate_lengthscale(train_t: jnp.array, train_y: jnp.array, tau: jnp.array, t0: float, tf: float, raw_sigf: jnp.array, raw_alpha: jnp.array, raw_beta: jnp.array, kernel_fn: Callable, warp_fn: Callable, gamma: float, len_candidates: Tuple[float] = (0.1, 0.5, 1.0, 10.0), n_splits: int = 5, key: Optional[jr.PRNGKey] = None) -> float:
    """
    5-fold cross-validation over lengthscale candidates
    Follows Course and Nair, 2023
    """
    n = len(train_t)
    if key is not None:
        idx = jr.permutation(key, n)
    else:
        idx = np.arange(n)
        np.random.shuffle(idx)
    idx = np.array(idx)
    fold_size = n // n_splits

    tc = train_t[:, None] if train_t.ndim == 1 else train_t
    tau_warped = warp_fn(tau, t0, tf, raw_alpha, raw_beta)

    best_len, best_err = len_candidates[0], float("inf")
    for li in len_candidates:
        raw_len_i = inverse_softplus(li)
        fold_errors = []
        for s in range(n_splits):
            val_idx = idx[s * fold_size : (s + 1) * fold_size]
            train_idx = np.concatenate([idx[:s * fold_size], idx[(s + 1) * fold_size:]])

            tc_train = tc[train_idx]
            tc_val = tc[val_idx]
            y_train = train_y[train_idx]
            y_val = train_y[val_idx]

            tc_train_w = warp_fn(tc_train, t0, tf, raw_alpha, raw_beta)
            tc_val_w = warp_fn(tc_val, t0, tf, raw_alpha, raw_beta)

            K_train = kernel_fn(tc_train_w, tau_warped, raw_len_i, raw_sigf)
            K_val = kernel_fn(tc_val_w, tau_warped, raw_len_i, raw_sigf)

            w = _solve_ls_svd(K_train, y_train, gamma=gamma)
            err = float(jnp.mean((K_val @ w - y_val) ** 2))
            fold_errors.append(err)

        mean_err = sum(fold_errors) / len(fold_errors)
        if mean_err < best_err:
            best_err = mean_err
            best_len = li
    return best_len


def _init_glm(n_tau: int, n_out: int, tau: jnp.array, t0: float, tf: float, len_init: float, sigf_init: float, alpha_init: float, beta_init: float, whitened: bool, kernel_fn: Callable, warp_fn: Callable, train_t: Optional[jnp.array] = None, train_y: Optional[jnp.array] = None, gamma: float = 1e-1, b_init: Optional[jnp.array] = None) -> Dict:
    """
    Initialize a single GLM (mean, eigenvalue, or orthogonal)
    """
    raw_len = inverse_softplus(len_init)
    raw_sigf = inverse_softplus(sigf_init)
    raw_alpha = inverse_softplus(alpha_init)
    raw_beta = inverse_softplus(beta_init)

    tau_warped = warp_fn(tau, t0, tf, raw_alpha, raw_beta)
    Ktt = kernel_fn(tau_warped, tau_warped, raw_len, raw_sigf)
    C = jnp.linalg.cholesky(Ktt + 1e-6 * jnp.eye(n_tau)) # the jitter of SVISE; part of the whitening, so it must not change once posteriors are stored

    # Fit weights to data if provided
    if train_t is not None and train_y is not None:
        tc = train_t[:, None] if train_t.ndim == 1 else train_t
        tc_warped = warp_fn(tc, t0, tf, raw_alpha, raw_beta)
        features = kernel_fn(tc_warped, tau_warped, raw_len, raw_sigf)
        w = _solve_ls_svd(features, train_y, gamma=gamma)
        raw_w = C.T @ w if whitened else w
    else:
        raw_w = jnp.zeros((n_tau, n_out))

    b = b_init if b_init is not None else jnp.zeros(n_out)

    return {
        "raw_w": raw_w,
        "b": b,
        "raw_len": raw_len,
        "raw_sigf": raw_sigf,
        "raw_alpha": raw_alpha,
        "raw_beta": raw_beta,
        "C": C,
    }


def init_gp_posterior(
    n_tau: int, #  number of inducing points
    K: int, # latent dimension
    t_span: Tuple[float, float], # the observation window (t_min, t_max)
    train_t: jnp.array, # (T) observation times
    train_y: jnp.array, # (T, K) observations
    len_init: Optional[float] = None, # initial kernel lengthscale (post-softplus), or None for cross-validation
    sigf_init: float = 1.0, # initial kernel signal std (post-softplus)
    alpha_init: float = 1.0, # initialization of Kumaraswamy warping parameter alpha
    beta_init: float = 1.0, # initialization of Kumaraswamy warping parameter beta
    full_cov: bool = True, # if True, spectral covariance with independent GLMs
    whitened: bool = True, # if True, use whitened parameterization
    kernel_fn: Callable = matern52_kernel, # kernel function
    warp_fn: Callable = kumaraswamy_warp, # kernel warping function
    gamma: float = 1e-1,
    key: Optional[jr.PRNGKey] = None, # random key for the lengthscale cross-validation; None uses the global numpy RNG
) -> Dict:
    """
    Initialize parameters for GPInferenceNetwork by fitting mean weights to data

    If len_init is None, the lengthscale is selected via 5-fold cross-validation over [0.1, 0.5, 1.0, 10.0]
    NOTE: to ensure reproducibility of fold split, pass a key whenever len_init is None

    When full_cov=True, three independent GLMs are created: mean, eigenvalues, and orthogonal, each with their own kernel hyperparameters
    """
    # Inducing points with margin
    nu = 1.0
    t0, tf = t_span[0] - nu, t_span[1] + nu
    tau = jnp.linspace(t0, tf, n_tau)[:, None]

    # Default hyperparams
    raw_sigf = inverse_softplus(sigf_init)
    raw_alpha = inverse_softplus(alpha_init)
    raw_beta = inverse_softplus(beta_init)

    # Cross-validate lengthscale if not provided
    if len_init is None:
        len_init = _cross_validate_lengthscale(
            train_t, train_y, tau, t0, tf,
            raw_sigf, raw_alpha, raw_beta,
            kernel_fn, warp_fn, gamma,
            key=key,
        )

    # Mean GLM (fitted to data)
    mean_glm = _init_glm(n_tau, K, tau, t0, tf, len_init, sigf_init, alpha_init, beta_init, whitened, kernel_fn, warp_fn, train_t=train_t, train_y=train_y, gamma=gamma)

    if full_cov:
        # Eigenvalue GLM: K outputs, bias chosen so softplus(bias) = 0.1
        eig_glm = _init_glm(n_tau, K, tau, t0, tf, 1.0, sigf_init, alpha_init, beta_init, whitened, kernel_fn, warp_fn, b_init=jnp.full(K, inverse_softplus(0.1)))
        # Orthogonal GLM: K*(K-1)/2 outputs, bias at 0 (identity rotation)
        n_skew = K * (K - 1) // 2
        orth_glm = _init_glm(n_tau, n_skew, tau, t0, tf, 1.0, sigf_init, alpha_init, beta_init, whitened, kernel_fn, warp_fn)
        return {
            "mean": mean_glm,
            "eigenvals": eig_glm,
            "orthogonal": orth_glm,
            "tau": tau,
            "t0": t0,
            "tf": tf,
        }
    else:
        # Diagonal: single covariance GLM with K outputs
        cov_glm = _init_glm(
            n_tau, K, tau, t0, tf, 1.0, sigf_init, alpha_init, beta_init,
            whitened, kernel_fn, warp_fn,
            b_init=jnp.full(K, inverse_softplus(0.1)),
        )
        return {
            "mean": mean_glm,
            "cov": cov_glm,
            "tau": tau,
            "t0": t0,
            "tf": tf,
        }


def _skew_sym_expm(v: jnp.array, K: int) -> jnp.array:
    """
    Construct a skew-symmetric matrix from K*(K-1)/2 parameters and
    return its matrix exponential (an orthogonal matrix)
    """
    idx = jnp.tril_indices(K, -1)
    S = jnp.zeros((K, K), dtype=v.dtype)
    S = S.at[idx].set(v)
    S = S - S.T
    return jax.scipy.linalg.expm(S)


class GPInferenceNetwork(Posterior):
    """
    GP-based posterior for latent SDEs using kernel GLM parameterization
    Matches SVISE's DiagonalMarginalSDE (full_cov=False) and SpectralMarginalSDE (full_cov=True)
    """
    K: int
    init_params: Dict
    full_cov: bool = True
    whitened: bool = True
    kernel_fn: Callable = matern52_kernel
    warp_fn: Callable = kumaraswamy_warp
    jitter: float = 1e-8

    def _glm_forward(self, glm_name: str, n_out: int, tc: jnp.array) -> jnp.array:
        """
        Evaluate a single GLM at query time tc
        """
        glm_init = self.init_params[glm_name]
        n_tau = self.init_params["tau"].shape[0]

        raw_w = self.param(f"{glm_name}_raw_w", lambda _, __: glm_init["raw_w"], (n_tau, n_out))
        b = self.param(f"{glm_name}_b", lambda _, __: glm_init["b"], (n_out,))
        raw_len = self.param(f"{glm_name}_raw_len", lambda _, __: glm_init["raw_len"], ())
        raw_sigf = self.param(f"{glm_name}_raw_sigf", lambda _, __: glm_init["raw_sigf"], ())
        raw_alpha = self.param(f"{glm_name}_raw_alpha", lambda _, __: glm_init["raw_alpha"], ())
        raw_beta = self.param(f"{glm_name}_raw_beta", lambda _, __: glm_init["raw_beta"], ())

        tau = self.init_params["tau"]
        t0, tf = self.init_params["t0"], self.init_params["tf"]

        # Unwhiten
        if self.whitened:
            C = jax.lax.stop_gradient(glm_init["C"])
            w = jax.scipy.linalg.solve_triangular(C.T, raw_w, lower=False)
        else:
            w = raw_w

        # Warp and evaluate kernel
        tc_warped = self.warp_fn(tc, t0, tf, raw_alpha, raw_beta)
        tau_warped = self.warp_fn(tau, t0, tf, raw_alpha, raw_beta)
        K_t_tau = self.kernel_fn(tc_warped, tau_warped, raw_len, raw_sigf)
        return (K_t_tau @ w + b)[0]

    @nn.compact
    def __call__(self, t: jnp.array, ctx: jnp.array, process_ctx: Callable[[jnp.array, jnp.array], jnp.array]):
        K = self.K
        tc = t[:, None] if t.ndim == 1 else t

        # Mean GLM
        m = self._glm_forward("mean", K, tc)

        if self.full_cov:
            # Eigenvalue GLM (independent kernel params)
            raw_eigs = self._glm_forward("eigenvals", K, tc)
            eigs = jax.nn.softplus(raw_eigs) + self.jitter

            # Orthogonal GLM (independent kernel params)
            n_skew = K * (K - 1) // 2
            raw_skew = self._glm_forward("orthogonal", n_skew, tc)
            U = _skew_sym_expm(raw_skew, K)

            R = U * jnp.sqrt(eigs)[None, :]  # U @ diag(sqrt(D))
        else:
            # Covariance GLM (independent kernel params)
            raw_eigs = self._glm_forward("cov", K, tc)
            eigs = jax.nn.softplus(raw_eigs) + self.jitter
            R = jnp.diag(jnp.sqrt(eigs))
        return m, R


def share_kernel_hyperparameters(params: Dict[str, Any], full_cov: bool = True) -> Dict[str, Any]:
    """
    Ties the kernel hyperparameters of a per-trial GP posterior across trials by broadcasting those of the first trial;
    used as the param_map of train
    """
    glm_names = ("mean", "eigenvals", "orthogonal") if full_cov else ("mean", "cov")
    shared = {f"{glm}_{suffix}" for glm in glm_names for suffix in ("raw_len", "raw_sigf", "raw_alpha", "raw_beta")}
    post_params = params["posterior_params"]["params"]
    post_params = {name: jnp.broadcast_to(value[0:1], value.shape) if name in shared else value for name, value in post_params.items()}
    return {**params, "posterior_params": {**params["posterior_params"], "params": post_params}}
