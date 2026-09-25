"""
Evaluation metrics, an iterated extended RTS smoother and plotting for the noisy Lorenz attractor experiment
"""

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
from sklearn.model_selection import KFold

from helmholtz_sde.sde import SDE
from helmholtz_sde.likelihood import Gaussian
from helmholtz_sde.posterior.encoder import ForwardGRUEncoder
from helmholtz_sde.posterior.nn_posterior import InferenceNetwork
from helmholtz_sde.posterior.drift import apply_inference_net_time_derivs


# --------------------- Lorenz dynamics ---------------------
def lorenz_drift(x: jnp.array, a: jnp.array) -> jnp.array:
    """
    Drift of the Lorenz system with parameters a = (a1, a2, a3), i.e. (sigma, rho, beta) in the usual notation
        dx1/dt = a1 (x2 - x1)
        dx2/dt = a2 x1 - x2 - x1 x3
        dx3/dt = x1 x2 - a3 x3
    """
    x1, x2, x3 = x
    a1, a2, a3 = a
    return jnp.array([a1 * (x2 - x1), a2 * x1 - x2 - x1 * x3, x1 * x2 - a3 * x3])


class LorenzAttractor(SDE):
    """
    The stochastic Lorenz attractor
    dx(t) = lorenz_drift(x(t), a) dt + G dw(t)
    """
    def __init__(self, K: int) -> None:
        if K != 3:
            raise ValueError(f"the Lorenz attractor is three-dimensional, got K={K}")
        super().__init__(K)

    def drift(self, x: jnp.array, t: float, sde_params: Dict[str, Any]) -> jnp.array:
        return lorenz_drift(x, sde_params["a"])


def compute_lorenz_fixed_points(means: np.ndarray, stds: np.ndarray, rho: float = 28.0, beta: float = 8.0 / 3.0) -> Dict[str, np.ndarray]:
    """
    The two non-trivial fixed points (-+r, -+r, rho - 1) with r = sqrt(beta (rho - 1)) of the Lorenz system, which are
    the centers of the two lobes of the attractor, in the standardized coordinates (x - means) / stds
    """
    r = np.sqrt(beta * (rho - 1.0))
    means, stds = np.asarray(means, dtype=float), np.asarray(stds, dtype=float)
    neg = (np.array([-r, -r, rho - 1.0]) - means) / stds
    pos = (np.array([+r, +r, rho - 1.0]) - means) / stds
    return {"neg": neg, "pos": pos}


# --------------------- Likelihood and posterior evaluation ---------------------
class GaussianFixedVariance(Gaussian):
    """
    Gaussian likelihood whose observation covariance is fixed to ssigma^2 I
    """
    def __init__(self, ssigma: float) -> None:
        super().__init__()
        self.ssigma = ssigma

    def _with_fixed_R(self, output_params: Dict[str, Any], D: int) -> Dict[str, Any]:
        return {**output_params, "R": self.ssigma ** 2 * jnp.eye(D)}

    def ll(self, x: jnp.array, y: jnp.array, t: float, output_params: Dict[str, Any]) -> jnp.array:
        return super().ll(x, y, t, self._with_fixed_R(output_params, y.shape[0]))

    def ell(self, y: jnp.array, t: float, mt: jnp.array, St: jnp.array, key: jr.PRNGKey, output_params: Dict[str, Any]) -> jnp.array:
        return super().ell(y, t, mt, St, key, self._with_fixed_R(output_params, y.shape[0]))


def posterior_true_logprob(ys_obs: jnp.array, obs_times: jnp.array, ys_true: jnp.array, t_grid: jnp.array, encoder: ForwardGRUEncoder, post_net: InferenceNetwork, params: Dict[str, Any], process_ctx: Callable, ssigma: float, batch_size: int = 64) -> float:
    """
    Log probability of the true latents under the approximate posterior, holding out training observations
    """
    B, D = ys_obs.shape[0], ys_obs.shape[-1]
    C, d = params["output_params"]["C"], params["output_params"]["d"]

    def _trial_logprobs(y_obs_b: jnp.array, obs_times_b: jnp.array, y_true_b: jnp.array) -> Tuple[jnp.array, jnp.array]:
        ctx_seq = encoder.apply(params["encoder_params"], y_obs_b)
        proc_ctx = partial(process_ctx, obs_times_b)

        def _at_time(t: float, y_t: jnp.array) -> jnp.array:
            t_arr = jnp.array([t], dtype=y_obs_b.dtype)
            mt, Rt, _, _ = apply_inference_net_time_derivs(post_net, params["posterior_params"], t_arr, ctx_seq, proc_ctx)
            mean_y = C @ mt + d
            cov_y = C @ (Rt @ Rt.T) @ C.T + ssigma ** 2 * jnp.eye(D)
            return tfd.MultivariateNormalFullCovariance(loc=mean_y, covariance_matrix=cov_y).log_prob(y_t)

        ll = vmap(_at_time)(t_grid, y_true_b) # (T_eval)
        not_obs = ~jnp.any(jnp.isclose(t_grid[:, None], obs_times_b[None, :]), axis=1) # (T_eval), True at held-out times
        return ll, not_obs

    vals, masks = [], []
    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        batch_vals, batch_mask = vmap(_trial_logprobs)(ys_obs[start:end], obs_times[start:end], ys_true[start:end])
        vals.append(np.asarray(batch_vals))
        masks.append(np.asarray(batch_mask))
    vals, masks = np.concatenate(vals, axis=0), np.concatenate(masks, axis=0) # (B, T_eval)
    return float(vals[masks].mean()) if masks.any() else float("nan") # NaN if every evaluation time is an observation time


# --------------------- Time-lagged correlations ---------------------
def _time_center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True) # center each time across trials


def _resolve_lags(max_lag: int, skip: int = 1) -> np.ndarray:
    return np.arange(0, max_lag + 1, skip, dtype=int)


def compute_lagged_autocorr(paths: np.ndarray, max_lag: int, skip: int = 1, start_idx: int = 0, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Time-lagged autocorrelation of each coordinate, estimated across trials at each time and then averaged over time
    NOTE: the dynamics are not assumed to be stationary
    """
    x = np.asarray(paths)[:, start_idx:, :]
    B, T, D = x.shape
    lags = _resolve_lags(max_lag, skip)

    xc = _time_center(x)
    var = np.mean(xc ** 2, axis=0) # (T, D) variance across trials at each time
    acorr = np.zeros((len(lags), D))
    for k, lag in enumerate(lags):
        cov = np.einsum("bti,bti->ti", xc[:, :T - lag, :], xc[:, lag:, :]) / B # (T - lag, D)
        corr_t = cov / np.sqrt(np.maximum(var[:T - lag] * var[lag:], eps))
        acorr[k] = np.mean(corr_t, axis=0) # mean over time
    return lags, acorr


def compute_lagged_crosscorr(paths: np.ndarray, lags: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Time-lagged cross-correlation matrices (n_lags, D, D), estimated across trials at each time and then averaged
    over time
    """
    x = np.asarray(paths)
    B, T, D = x.shape

    xc = _time_center(x)
    std = np.maximum(np.sqrt(np.mean(xc ** 2, axis=0)), eps) # (T, D) standard deviation across trials at each time
    xc_left = np.ascontiguousarray(xc.transpose(1, 2, 0)) # (T, D, B)
    xc_right = np.ascontiguousarray(xc.transpose(1, 0, 2)) # (T, B, D)
    crosscorr = np.zeros((len(lags), D, D))
    for k, lag in enumerate(lags):
        cov = np.matmul(xc_left[:T - lag], xc_right[lag:]) / B # (T - lag, D, D), one batched product over trials
        corr_t = cov / (std[:T - lag, :, None] * std[lag:, None, :])
        crosscorr[k] = np.mean(corr_t, axis=0) # mean over time
    return crosscorr


def plot_lagged_autocorr_compare(paths_true: np.ndarray, paths_model: np.ndarray, max_lag: int, skip: int = 1, start_idx: int = 0, dt: Optional[float] = None, fontsize: float = 12) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plots the time-lagged autocorrelation of each coordinate under the true and the learned prior
    """
    lags, ac_true = compute_lagged_autocorr(paths_true, max_lag, skip, start_idx)
    _, ac_model = compute_lagged_autocorr(paths_model, max_lag, skip, start_idx)
    D = ac_true.shape[1]
    xvals, xlabel = (lags, "lag") if dt is None else (dt * lags, "time lag")

    fig, axs = plt.subplots(1, D, figsize=(5 * D, 4), squeeze=False)
    for d in range(D):
        ax = axs[0, d]
        ax.plot(xvals, ac_true[:, d], linewidth=2, label="true prior")
        ax.plot(xvals, ac_model[:, d], linewidth=2, linestyle="--", label="learned prior")
        ax.set_title(f"Lagged autocorr, dim {d}", fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel("correlation", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 2)
        ax.grid(alpha=0.25)
        if d == 0:
            ax.legend(fontsize=fontsize)
    fig.tight_layout()
    return fig, axs


def global_correlation_error(paths_true: np.ndarray, paths_model: np.ndarray, dt: float, max_time: float = 2.0, skip: int = 50) -> float:
    """
    Squared Frobenius distance between the time-lagged cross-correlation matrices of the true and the learned prior,
    averaged over lags 0, skip, ..., max_time / dt
    """
    lags = _resolve_lags(int(round(max_time / dt)), skip)
    corr_true = compute_lagged_crosscorr(paths_true, lags)
    corr_model = compute_lagged_crosscorr(paths_model, lags)
    return float(np.mean(np.sum((corr_true - corr_model) ** 2, axis=(1, 2))))


# --------------------- Within-lobe time-lagged correlations ---------------------
def assign_lobes_by_nearest_center(x: np.ndarray, centers: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Labels each time point of each path with the closer lobe center (0 for negative, 1 for positive)
    """
    d_neg = np.sum((x - centers["neg"][None, None, :]) ** 2, axis=-1)
    d_pos = np.sum((x - centers["pos"][None, None, :]) ** 2, axis=-1)
    return np.where(d_pos < d_neg, 1, 0) # (B, T)


def fit_plane_from_true_lobe(x_true: np.ndarray, labels_true: np.ndarray, center: np.ndarray, lobe_label: int, fit_radius_quantile: float = 0.95) -> np.ndarray:
    """
    Fits a plane through the given center to the true samples assigned to one lobe by PCA, using only the samples
    within the fit_radius_quantile quantile of the distance to the center
    """
    X = x_true[labels_true == lobe_label] - center[None, :] # samples in this lobe, relative to its center
    rad = np.linalg.norm(X, axis=1)
    X = X[rad <= np.quantile(rad, fit_radius_quantile)]
    _, _, vh = np.linalg.svd(X, full_matrices=False) # the leading right singular vectors span the plane
    return vh[:2].T


def project_to_plane(x: np.ndarray, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """
    Projects paths (B, T, D) onto the plane with the given center and basis, returning coordinates (B, T, 2)
    """
    return np.einsum("btd,dk->btk", x - center[None, None, :], basis)


def extract_contiguous_segments_for_lobe(z: np.ndarray, labels: np.ndarray, lobe_label: int, min_count: int = 50) -> List[np.ndarray]:
    """
    Extracts the maximal contiguous visits of at least min_count time points to one lobe from the projected paths
    """
    segments = []
    for b in range(z.shape[0]):
        idx = np.flatnonzero(labels[b] == lobe_label) # time points in this lobe
        for block in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1): # split at gaps into contiguous visits
            if block.size >= min_count:
                segments.append(z[b, block, :])
    return segments


def lagged_correlation_matrix_from_segments(segments: List[np.ndarray], lags: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Time-lagged 2 x 2 correlation matrices (n_lags, 2, 2) of the projected lobe visits, with time measured from the
    start of each visit; NaN at lags that no visit is long enough for
    """
    C = np.full((len(lags), 2, 2), np.nan, dtype=float)
    if len(segments) == 0:
        return C

    # Visits zero-padded to a common length, with a mask of the times each visit reaches
    lengths = np.array([seg.shape[0] for seg in segments])
    max_T = int(lengths.max())
    Z = np.zeros((len(segments), max_T, 2), dtype=float)
    for s, seg in enumerate(segments):
        Z[s, :seg.shape[0]] = seg
    valid = np.arange(max_T)[None, :] < lengths[:, None] # (n_visits, max_T)
    count = valid.sum(axis=0) # (max_T) number of visits reaching each time

    # Per-time means and standard deviations, pooled over the visits that reach each time
    mean = Z.sum(axis=0) / count[:, None]
    Zc = (Z - mean[None]) * valid[:, :, None] # centered, zero beyond the end of each visit
    std = np.maximum(np.sqrt(np.sum(Zc ** 2, axis=0) / count[:, None]), eps)

    # Lagged correlations at each time as one batched product over the visits
    Zc_left = np.ascontiguousarray(Zc.transpose(1, 2, 0)) # (max_T, 2, n_visits)
    Zc_right = np.ascontiguousarray(Zc.transpose(1, 0, 2)) # (max_T, n_visits, 2)
    for k, lag in enumerate(lags):
        if lag >= max_T:
            continue
        cov = np.matmul(Zc_left[:max_T - lag], Zc_right[lag:]) / count[lag:, None, None] # (max_T - lag, 2, 2)
        denom = np.maximum(std[:max_T - lag, :, None] * std[lag:, None, :], eps)
        C[k] = np.mean(cov / denom, axis=0)
    return C


def compute_lorenz_rotation_summaries(paths_true: np.ndarray, paths_model: np.ndarray, centers: Dict[str, np.ndarray], min_count: int = 50, max_lag: int = 300, skip: int = 5, fit_radius_quantile: float = 0.95) -> Dict[str, Any]:
    """
    Compares the rotation within each lobe of the attractor under the true and the learned prior
    """
    x_true, x_model = np.asarray(paths_true), np.asarray(paths_model)
    centers = {name: np.asarray(centers[name], dtype=float) for name in ("neg", "pos")}
    lags = _resolve_lags(max_lag, skip)

    # Label each time point with the nearer lobe center
    labels_true = assign_lobes_by_nearest_center(x_true, centers)
    labels_model = assign_lobes_by_nearest_center(x_model, centers)

    results = {"lags": lags, "lobes": {}}
    for lobe_name, lobe_label in [("neg", 0), ("pos", 1)]:
        center = centers[lobe_name]

        # Project onto the plane fitted to the true samples in this lobe
        basis = fit_plane_from_true_lobe(x_true, labels_true, center, lobe_label, fit_radius_quantile)
        z_true = project_to_plane(x_true, center, basis)
        z_model = project_to_plane(x_model, center, basis)

        # Time-lagged correlations of the contiguous visits to this lobe
        segs_true = extract_contiguous_segments_for_lobe(z_true, labels_true, lobe_label, min_count)
        segs_model = extract_contiguous_segments_for_lobe(z_model, labels_model, lobe_label, min_count)
        results["lobes"][lobe_name] = {"Corr_true": lagged_correlation_matrix_from_segments(segs_true, lags), "Corr_model": lagged_correlation_matrix_from_segments(segs_model, lags)}
    return results


def plot_lobe_lagged_correlation_matrices(results: Dict[str, Any], dt: Optional[float] = None, figsize: Tuple[float, float] = (18, 8), fontsize: float = 12) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plots the entries of the within-lobe time-lagged correlation matrices returned by compute_lorenz_rotation_summaries
    """
    fig, axs = plt.subplots(2, 4, figsize=figsize, constrained_layout=True)
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    xvals, xlabel = (results["lags"], "lag") if dt is None else (dt * results["lags"], "time lag")
    for row, (lobe_name, title_prefix) in enumerate([("neg", "negative lobe"), ("pos", "positive lobe")]):
        res = results["lobes"][lobe_name]
        for col, (i, j) in enumerate(pairs):
            ax = axs[row, col]
            ax.plot(xvals, res["Corr_true"][:, i, j], linewidth=2, label="true prior")
            ax.plot(xvals, res["Corr_model"][:, i, j], linewidth=2, linestyle="--", label="learned prior")
            ax.set_title(f"{title_prefix}: Corr[{i},{j}](τ)", fontsize=fontsize)
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.set_ylabel("correlation", fontsize=fontsize)
            ax.tick_params(labelsize=fontsize - 2)
            ax.grid(alpha=0.25)
            if row == 0 and col == 0:
                ax.legend(fontsize=fontsize)
    return fig, axs


def within_lobe_correlation_error(paths_true: np.ndarray, paths_model: np.ndarray, centers: Dict[str, np.ndarray], dt: float, max_time: float = 2.0, skip: int = 50, min_count: int = 50, fit_radius_quantile: float = 0.95) -> float:
    """
    Squared Frobenius distance between the within-lobe time-lagged correlation matrices of the true and the learned
    prior, averaged over lags 0, skip, ..., max_time / dt and over the two lobes
    """
    summaries = compute_lorenz_rotation_summaries(paths_true, paths_model, centers, min_count, int(round(max_time / dt)), skip, fit_radius_quantile)
    errs = []
    for lobe_name in ["neg", "pos"]:
        C_true, C_model = summaries["lobes"][lobe_name]["Corr_true"], summaries["lobes"][lobe_name]["Corr_model"]
        errs.append(np.mean(np.sum((C_true - C_model) ** 2, axis=(1, 2))))
    return float(np.mean(errs))


# --------------------- Marginal KL divergence ---------------------
def _make_bandwidth_grid(xs: np.ndarray, n_grid: int = 15) -> np.ndarray:
    """
    Geometric grid of KDE bandwidths, scaled by the mean per-coordinate standard deviation of the samples
    """
    scale = max(float(np.mean(np.std(xs, axis=0, ddof=1))), 1e-3)
    return np.geomspace(max(0.05 * scale, 1e-4), 2.0 * scale, num=n_grid)


@jax.jit
def _kde_log_density(x_query: jnp.array, x_train: jnp.array, bandwidths: jnp.array) -> jnp.array:
    """
    Log density of the Gaussian kernel density estimate with samples x_train (N, D) at the points x_query (M, D), for
    each bandwidth in bandwidths (n_h), returned as an (n_h, M) array
    """
    N, D = x_train.shape
    d2 = jnp.sum(jnp.square(x_query[:, None, :] - x_train[None, :, :]), axis=-1) # (M, N)
    log_kernels = jsp.special.logsumexp(-d2[None] / (2 * bandwidths[:, None, None] ** 2), axis=-1) # (n_h, M)
    return log_kernels - jnp.log(N) - 0.5 * D * jnp.log(2 * jnp.pi * bandwidths[:, None] ** 2)


def _select_kde_bandwidth_cv(xs: np.ndarray, bandwidth_grid: np.ndarray, n_splits: int = 3, random_state: int = 0) -> float:
    """
    Selects the KDE bandwidth by K-fold cross-validation, maximizing the held-out log likelihood
    """
    scores = jnp.zeros(len(bandwidth_grid))
    for train_idx, val_idx in KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(xs):
        scores = scores + jnp.mean(_kde_log_density(xs[val_idx], xs[train_idx], bandwidth_grid), axis=1)
    return float(bandwidth_grid[int(jnp.argmax(scores))])


def _estimate_single_time_kl(x_true: np.ndarray, x_model: np.ndarray, n_bandwidth_splits: int = 3, n_outer_splits: int = 10, random_state: int = 0) -> float:
    """
    Estimates KL(P || Q) at one time, where P and Q are KDEs fit to the true and the model samples, as the average of
    log p - log q over held-out true samples across n_outer_splits folds
    """
    bw_model = jnp.array([_select_kde_bandwidth_cv(x_model, _make_bandwidth_grid(x_model), n_bandwidth_splits, random_state)])
    bandwidth_grid_true = _make_bandwidth_grid(x_true)

    rs = random_state + 1
    kl_vals = []
    for fold_idx, (train_idx, test_idx) in enumerate(KFold(n_splits=n_outer_splits, shuffle=True, random_state=rs).split(x_true)):
        bw_true = jnp.array([_select_kde_bandwidth_cv(x_true[train_idx], bandwidth_grid_true, n_bandwidth_splits, rs + 20000 + fold_idx)])
        log_p = _kde_log_density(x_true[test_idx], x_true[train_idx], bw_true)[0]
        log_q = _kde_log_density(x_true[test_idx], x_model, bw_model)[0]
        kl_vals.append(float(jnp.mean(log_p - log_q)))
    return float(np.mean(kl_vals))


def compute_marginal_kde_kl(ys_test: np.ndarray, ys_eval: np.ndarray, times: Sequence[int], n_bandwidth_splits: int = 3, n_outer_splits: int = 10, random_state: int = 0) -> float:
    """
    Marginal KL divergence KL(P_t || Q_t) between the true and the learned prior, estimated by kernel density
    estimation at each of the given times and averaged over them
    """
    ys_test, ys_eval = np.asarray(ys_test), np.asarray(ys_eval)
    kls = [_estimate_single_time_kl(ys_test[:, t, :], ys_eval[:, t, :], n_bandwidth_splits, n_outer_splits, random_state + 1000 * int(t)) for t in times]
    return float(np.mean(kls))


# --------------------- Summary statistics ---------------------
def summary_stats(vals: Sequence[float], prefix: str) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {f"{prefix}_mean": float(np.mean(arr)), f"{prefix}_std": std, f"{prefix}_median": float(np.median(arr))}


# --------------------- Iterated extended RTS smoother ---------------------
# Approximate Gaussian smoother of the true Lorenz dynamics in observation space, used as a reference posterior
def _symmetrize(M: jnp.array) -> jnp.array:
    return 0.5 * (M + jnp.swapaxes(M, -1, -2))


def _solve_spd_batched(A: jnp.array, rhs: jnp.array, jitter: float = 1e-8) -> jnp.array:
    D = A.shape[-1]
    A = _symmetrize(A) + jitter * jnp.eye(D, dtype=A.dtype)
    L = jnp.linalg.cholesky(A)
    Y = jsp.linalg.solve_triangular(L, rhs, lower=True)
    return jsp.linalg.solve_triangular(jnp.swapaxes(L, -1, -2), Y, lower=False)


def make_y_dynamics(a: jnp.array, C_true: jnp.array, d_true: jnp.array, sigma: float) -> Tuple[Callable[[jnp.array], jnp.array], jnp.array]:
    """
    Drift and diffusion coefficient of the Lorenz dynamics in observation space y = C_true x + d_true, where the
    latent dynamics have diffusion coefficient sigma I
    """
    C_inv = jnp.linalg.inv(C_true)
    G_y = sigma * C_true

    def drift_y(y: jnp.array) -> jnp.array:
        x = C_inv @ (y - d_true)
        return C_true @ lorenz_drift(x, a)
    return drift_y, G_y


def build_dense_observations_batch(ys_obs_batch: jnp.array, obs_idx: jnp.array, T: int) -> Tuple[jnp.array, jnp.array]:
    """
    Places the observations on the dense time grid, returning y_dense (B, T, D), zero at unobserved times, and the
    boolean mask obs_mask (B, T) of observed times
    """
    B, _, D = ys_obs_batch.shape
    b_idx = jnp.arange(B)[:, None]
    y_dense = jnp.zeros((B, T, D), dtype=ys_obs_batch.dtype).at[b_idx, obs_idx, :].set(ys_obs_batch)
    obs_mask = jnp.zeros((B, T), dtype=bool).at[b_idx, obs_idx].set(True)
    return y_dense, obs_mask


def interpolate_obs_to_hidden_grid_batch(ys_obs_batch: jnp.array, obs_idx: jnp.array, T: int) -> jnp.array:
    """
    Linearly interpolates the observations onto the dense time grid, to initialize the smoother means
    """
    D = ys_obs_batch.shape[-1]
    full_idx = jnp.arange(T)

    def interp_one(y_obs_i: jnp.array, obs_idx_i: jnp.array) -> jnp.array:
        return jnp.stack([jnp.interp(full_idx, obs_idx_i, y_obs_i[:, j]) for j in range(D)], axis=-1)
    return jax.vmap(interp_one)(ys_obs_batch, obs_idx)


def linearize_transition_along_path_yspace_batch(ref_path: jnp.array, dt: float, a: jnp.array, C_true: jnp.array, d_true: jnp.array, sigma: float) -> Tuple[jnp.array, jnp.array, jnp.array]:
    """
    Linearizes the Euler-Maruyama transition y_{t+1} = y_t + dt drift_y(y_t) + noise along the reference path
    Returns the transition matrices F (B, T-1, D, D) and offsets b (B, T-1, D) with y_{t+1} ~ N(F_t y_t + b_t, Q), and
    the transition covariance Q (D, D)
    """
    drift_y, G_y = make_y_dynamics(a, C_true, d_true, sigma)

    def g(y: jnp.array) -> jnp.array:
        return y + dt * drift_y(y)

    y_ref = ref_path[:, :-1, :] # (B, T-1, D)
    F = jax.vmap(jax.vmap(jax.jacfwd(g)))(y_ref) # (B, T-1, D, D)
    g_ref = jax.vmap(jax.vmap(g))(y_ref) # (B, T-1, D)
    b = g_ref - jnp.einsum("btij,btj->bti", F, y_ref)
    Q = _symmetrize(dt * (G_y @ G_y.T))
    return F, b, Q


def identity_obs_update_batch(m_pred: jnp.array, P_pred: jnp.array, y_obs: jnp.array, obs_mask: jnp.array, R: jnp.array, jitter: float = 1e-8) -> Tuple[jnp.array, jnp.array]:
    """
    Kalman filter update for the observation model z_t = y_t + r_t with r_t ~ N(0, R);
    trials with obs_mask False keep their predicted moments
    """
    B, D = m_pred.shape
    I_B = jnp.broadcast_to(jnp.eye(D, dtype=m_pred.dtype), (B, D, D))
    R_B = jnp.broadcast_to(R, (B, D, D))

    S = _symmetrize(P_pred + R_B)
    K = jnp.swapaxes(_solve_spd_batched(S, jnp.swapaxes(P_pred, -1, -2), jitter=jitter), -1, -2) # Kalman gain
    m_filt_obs = m_pred + jnp.einsum("bij,bj->bi", K, y_obs - m_pred)
    IK = I_B - K
    P_filt_obs = _symmetrize(IK @ P_pred @ jnp.swapaxes(IK, -1, -2) + K @ R_B @ jnp.swapaxes(K, -1, -2)) # Joseph form

    m_filt = jnp.where(obs_mask[:, None], m_filt_obs, m_pred)
    P_filt = jnp.where(obs_mask[:, None, None], P_filt_obs, P_pred)
    return m_filt, P_filt


def rts_smoother_linear_time_varying_identity_obs_batch(F: jnp.array, b: jnp.array, Q: jnp.array, y0_mean: jnp.array, y0_cov: jnp.array, y_dense: jnp.array, obs_mask: jnp.array, R: jnp.array, jitter: float = 1e-8) -> Tuple[jnp.array, jnp.array]:
    """
    Linear Gaussian RTS smoother for the time-varying model y_{t+1} ~ N(F_t y_t + b_t, Q), y_0 ~ N(y0_mean, y0_cov),
    with identity observations at the masked times; batched over trials and scanned over time
    Returns the smoothed means (B, T, D) and covariances (B, T, D, D)
    """
    B, T, D = y_dense.shape
    Q_B = jnp.broadcast_to(Q, (B, D, D))

    # Filtering state at t = 0
    m_pred0 = jnp.broadcast_to(y0_mean, (B, D))
    P_pred0 = jnp.broadcast_to(y0_cov, (B, D, D))
    m_filt0, P_filt0 = identity_obs_update_batch(m_pred0, P_pred0, y_dense[:, 0, :], obs_mask[:, 0], R, jitter=jitter)

    # Forward pass (Kalman filter)
    def fwd_step(carry, inputs):
        m_filt_prev, P_filt_prev = carry
        Ft, bt, y_t, mask_t = inputs
        m_pred = jnp.einsum("bij,bj->bi", Ft, m_filt_prev) + bt
        P_pred = _symmetrize(Ft @ P_filt_prev @ jnp.swapaxes(Ft, -1, -2) + Q_B)
        m_filt, P_filt = identity_obs_update_batch(m_pred, P_pred, y_t, mask_t, R, jitter=jitter)
        return (m_filt, P_filt), (m_pred, P_pred, m_filt, P_filt)

    _, fwd_out = jax.lax.scan(
        fwd_step,
        (m_filt0, P_filt0),
        (
            jnp.swapaxes(F, 0, 1), # (T-1, B, D, D)
            jnp.swapaxes(b, 0, 1), # (T-1, B, D)
            jnp.swapaxes(y_dense[:, 1:, :], 0, 1), # (T-1, B, D)
            jnp.swapaxes(obs_mask[:, 1:], 0, 1), # (T-1, B)
        ),
    )
    m_pred_rest, P_pred_rest, m_filt_rest, P_filt_rest = fwd_out

    # Prepend t = 0 and move time back to axis 1
    m_pred = jnp.concatenate([m_pred0[:, None, :], jnp.swapaxes(m_pred_rest, 0, 1)], axis=1)
    P_pred = jnp.concatenate([P_pred0[:, None, :, :], jnp.swapaxes(P_pred_rest, 0, 1)], axis=1)
    m_filt = jnp.concatenate([m_filt0[:, None, :], jnp.swapaxes(m_filt_rest, 0, 1)], axis=1)
    P_filt = jnp.concatenate([P_filt0[:, None, :, :], jnp.swapaxes(P_filt_rest, 0, 1)], axis=1)

    # Backward pass (RTS smoother)
    mT, PT = m_filt[:, -1, :], P_filt[:, -1, :, :]

    def bwd_step(carry, inputs):
        m_next_s, P_next_s = carry
        m_f_t, P_f_t, m_p_next, P_p_next, F_t = inputs
        P_p_next_inv = _solve_spd_batched(P_p_next, jnp.broadcast_to(jnp.eye(D, dtype=P_p_next.dtype), P_p_next.shape), jitter=jitter)
        J_t = P_f_t @ jnp.swapaxes(F_t, -1, -2) @ P_p_next_inv # smoother gain
        m_s_t = m_f_t + jnp.einsum("bij,bj->bi", J_t, (m_next_s - m_p_next))
        P_s_t = _symmetrize(P_f_t + J_t @ (P_next_s - P_p_next) @ jnp.swapaxes(J_t, -1, -2))
        return (m_s_t, P_s_t), (m_s_t, P_s_t)

    _, (m_s_rev, P_s_rev) = jax.lax.scan(
        bwd_step,
        (mT, PT),
        (
            jnp.swapaxes(m_filt[:, :-1, :], 0, 1)[::-1],
            jnp.swapaxes(P_filt[:, :-1, :, :], 0, 1)[::-1],
            jnp.swapaxes(m_pred[:, 1:, :], 0, 1)[::-1],
            jnp.swapaxes(P_pred[:, 1:, :, :], 0, 1)[::-1],
            jnp.swapaxes(F, 0, 1)[::-1],
        ),
    )

    # Undo the time reversal, append t = T-1 and move time back to axis 1
    m_smooth = jnp.concatenate([jnp.swapaxes(m_s_rev[::-1], 0, 1), mT[:, None, :]], axis=1)
    P_smooth = jnp.concatenate([jnp.swapaxes(P_s_rev[::-1], 0, 1), PT[:, None, :, :]], axis=1)
    return m_smooth, P_smooth


@partial(jax.jit, static_argnames=("n_iters",))
def run_ie_rts_smoother_batch_fast(ys_obs_batch: jnp.array, obs_idx: jnp.array, t_grid: jnp.array, a: jnp.array, C_true: jnp.array, d_true: jnp.array, sigma: float, ssigma: float, n_iters: int = 10, jitter: float = 1e-8) -> Dict[str, jnp.array]:
    """
    Iterated extended RTS smoother for the true Lorenz dynamics in observation space, batched over trials
    Returns the smoothed means m (B, T, D) and covariances S (B, T, D, D) of the final iteration
    """
    if n_iters < 1:
        raise ValueError(f"n_iters must be positive, got {n_iters}")
    B, n_obs, D = ys_obs_batch.shape
    T = t_grid.shape[0]
    dt = t_grid[1] - t_grid[0]
    obs_idx = jnp.broadcast_to(obs_idx, (B, n_obs)) # observation indices may be shared across trials
    y_dense, obs_mask = build_dense_observations_batch(ys_obs_batch, obs_idx, T)

    # Prior in observation space: y(0) = C_true x(0) + d_true with x(0) ~ N(0, I)
    y0_mean = d_true
    y0_cov = C_true @ C_true.T
    R = (ssigma ** 2) * jnp.eye(D, dtype=ys_obs_batch.dtype)

    def _smooth(ref_path: jnp.array) -> Tuple[jnp.array, jnp.array]:
        F, b, Q = linearize_transition_along_path_yspace_batch(ref_path, dt, a, C_true, d_true, sigma)
        return rts_smoother_linear_time_varying_identity_obs_batch(F, b, Q, y0_mean, y0_cov, y_dense, obs_mask, R, jitter)

    # NOTE: the scan carries only the reference path, so memory does not grow with n_iters
    ref_path0 = interpolate_obs_to_hidden_grid_batch(ys_obs_batch, obs_idx, T)
    ref_path, _ = jax.lax.scan(lambda path, _: (_smooth(path)[0], None), ref_path0, xs=None, length=n_iters - 1)
    m, S = _smooth(ref_path)
    return {"m": m, "S": S}


def plot_rts_posterior_marginals(ys_obs: jnp.array, obs_times: jnp.array, ms_rts: jnp.array, Ss_rts: jnp.array, true_latents: Optional[jnp.array] = None, t_max: float = 1.0, num_std: float = 1.0, fontsize: float = 12.0) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plots the smoothed marginals of each coordinate (mean and +-num_std standard deviations) for every trial, together
    with the observations and, if given, the true paths
    """
    B, T, D = ms_rts.shape
    ts = np.linspace(0.0, t_max, T)
    means = np.asarray(ms_rts)
    stds = np.asarray(jnp.sqrt(jnp.maximum(jnp.diagonal(Ss_rts, axis1=-2, axis2=-1), 1e-8))) # (B, T, D)
    ys, ot = np.asarray(ys_obs), np.asarray(obs_times)
    if true_latents is not None:
        true_latents = np.asarray(true_latents)
        ts_true = np.linspace(0.0, float(t_max), true_latents.shape[1])

    labels = [rf"$\boldsymbol{{x}}({i})$" for i in range(D)]
    ncols = D if D <= 3 else 2
    nrows = int(np.ceil(D / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows), sharex=True)
    axs = np.array(axs).reshape(-1)
    for j in range(D):
        ax = axs[j]
        for b in range(B):
            mu, sd = means[b, :, j], stds[b, :, j]
            (line,) = ax.plot(ts, mu, linewidth=2.0)
            ax.fill_between(ts, mu - num_std * sd, mu + num_std * sd, alpha=0.25, color=line.get_color())
            ax.plot(ot[b], ys[b, :, j], linestyle="None", marker="x", markersize=6, mew=1.5, color=line.get_color())
            if true_latents is not None:
                ax.plot(ts_true[::10], true_latents[b, ::10, j], color="0.6", linewidth=1.5, zorder=1) # subsampled for speed
        ax.set_ylabel(labels[j], fontsize=fontsize)
        ax.set_xlabel("t", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 2)
    for ax_j in range(D, len(axs)):
        fig.delaxes(axs[ax_j])
    fig.tight_layout()
    return fig, axs[:D]
