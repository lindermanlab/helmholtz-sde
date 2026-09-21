"""
Helpers for the 2D cylinder flow experiment
"""

import math
from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax import lax

from helmholtz_sde.sde import SDE
from helmholtz_sde.utils.general_helpers import simulate_sde


# --------------------- Simulation ---------------------
def simulate_frames(key: jr.PRNGKey, x0: jnp.array, sde: SDE, sde_params: Dict[str, Any], dt_frame: float, n_frames: int, steps_per_frame: int) -> jnp.array:
    """
    Euler-Maruyama path of the SDE started at x0, stored only at the frame times dt_frame, ..., n_frames dt_frame
    (n_frames, K); the SDE is assumed time-homogeneous
    """
    def _frame(x: jnp.array, key_frame: jr.PRNGKey) -> Tuple[jnp.array, jnp.array]:
        x_next = simulate_sde(key_frame, x, sde, sde_params, t_max=dt_frame, n_timesteps=steps_per_frame)[-1]
        return x_next, x_next

    return lax.scan(_frame, x0, jr.split(key, n_frames))[1]


# --------------------- Decoding and correlations ---------------------
def decode_velocity(z: np.ndarray, pca_components: np.ndarray, pca_mean: np.ndarray, grid_shape: Tuple[int, int], z_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Velocity components u and v (..., n_y, n_x) of the POD coefficients z (..., K), which were rescaled by z_scale
    """
    state = (np.asarray(z) * z_scale) @ pca_components + pca_mean # (..., 2 n_y n_x)
    n_spatial = grid_shape[0] * grid_shape[1]
    return state[..., :n_spatial].reshape(*state.shape[:-1], *grid_shape), state[..., n_spatial:].reshape(*state.shape[:-1], *grid_shape)


def spacetime_correlation(signal: np.ndarray, dx: float, dt: float, max_lag_x: float, max_lag_t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-point space-time correlation coefficient C(dx, tau) of a signal (T, X) on a uniform grid, estimated by FFT
    with the unbiased (overlap-normalized) estimator, for lags |dx| <= max_lag_x and 0 <= tau <= max_lag_t
    """
    n_t, n_x = signal.shape
    pad_t, pad_x = 2 * n_t - 1, 2 * n_x - 1 # zero padding for a linear correlation
    F = np.fft.rfft2(signal, s=(pad_t, pad_x))
    R = np.fft.irfft2(F * np.conj(F), s=(pad_t, pad_x))
    overlap = np.concatenate([np.arange(n_t, 0, -1), np.arange(1, n_t)])[:, None] * np.concatenate([np.arange(n_x, 0, -1), np.arange(1, n_x)])[None, :] # pairs at each lag
    C = R / overlap / (R[0, 0] / overlap[0, 0])
    n_tau, n_dx = min(n_t, round(max_lag_t / dt) + 1), min(n_x - 1, round(max_lag_x / dx))
    C = np.fft.fftshift(C[:n_tau], axes=1)[:, pad_x // 2 - n_dx:pad_x // 2 + n_dx + 1] # positive time lags, spatial lags centred at zero
    return C, np.arange(-n_dx, n_dx + 1) * dx, np.arange(n_tau) * dt


# --------------------- Plotting ---------------------
def plot_pod_modes(t_true: Optional[np.ndarray] = None, z_true: Optional[np.ndarray] = None, t_model: Optional[np.ndarray] = None, mean: Optional[np.ndarray] = None, band: Optional[Tuple[np.ndarray, np.ndarray]] = None, samples: Optional[np.ndarray] = None, n_modes: int = 10, ncols: int = 2, fontsize: float = 12., xlabel: str = "Time", ylabel: str = "POD") -> Tuple[plt.Figure, np.ndarray]:
    """
    Grid of the first n_modes POD coefficients over time
    """
    nrows = math.ceil(n_modes / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 2.0 * nrows), sharex=True, squeeze=False)
    for j, ax in enumerate(axs.ravel()[:n_modes]):
        if z_true is not None:
            ax.plot(t_true, np.asarray(z_true)[:, j], color="0.6", linewidth=1.0)
        if samples is not None:
            ax.plot(t_model, np.asarray(samples)[:, :, j].T, color="C0", linewidth=0.6, alpha=0.3)
        if band is not None:
            ax.fill_between(t_model, np.asarray(band[0])[:, j], np.asarray(band[1])[:, j], color="C0", alpha=0.25)
        if mean is not None:
            ax.plot(t_model, np.asarray(mean)[:, j], color="C0", linewidth=1.5)
        ax.set_ylabel(f"{ylabel} {j + 1}", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 2)
    for ax in axs.ravel()[n_modes:]:
        fig.delaxes(ax)
    for ax in axs[-1]:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    fig.tight_layout()
    return fig, axs


def plot_field(field: np.ndarray, x_coord: np.ndarray, y_coord: np.ndarray, ax: plt.Axes, u: Optional[np.ndarray] = None, v: Optional[np.ndarray] = None, cmap: str = "Spectral_r", vmin: Optional[float] = None, vmax: Optional[float] = None, title: Optional[str] = None, fontsize: float = 12.):
    """
    Scalar field (n_y, n_x) over the flow domain with the cylinder and, if given, the streamlines of the velocity (u, v)
    """
    im = ax.pcolormesh(x_coord, y_coord, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
    if u is not None:
        ax.streamplot(np.unique(x_coord), np.unique(y_coord), u, v, color="k", density=2, linewidth=0.5, arrowsize=0.8)
    ax.add_patch(plt.Circle((0, 0), 0.5, facecolor="grey", edgecolor="k", linewidth=0.8, zorder=10))
    ax.set_xlim(x_coord.min(), x_coord.max()) # streamplot can extend the automatic limits
    ax.set_ylim(y_coord.min(), y_coord.max())
    ax.set_aspect("equal")
    ax.tick_params(labelsize=fontsize - 2)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    return im


def add_colorbar(fig: plt.Figure, im, axs, label: str, fontsize: float = 12.):
    cbar = fig.colorbar(im, ax=axs, location="right", shrink=0.9, pad=0.01)
    cbar.set_label(label, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize - 2)
    return cbar


def plot_spacetime_correlation(C: np.ndarray, lag_x: np.ndarray, lag_t: np.ndarray, ax: plt.Axes, title: Optional[str] = None, xlabel: Optional[str] = r"$\tau$", ylabel: Optional[str] = r"$\Delta x_1$", fontsize: float = 12., tick_length: float = 6., tick_width: float = 1., spine_width: float = 1.):
    """
    Space-time correlation with the time lag on the horizontal axis and the space lag on the vertical axis
    """
    im = ax.pcolormesh(lag_t, lag_x, C.T, cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="gouraud", rasterized=True)
    ax.set_xlim(lag_t[0], lag_t[-1])
    ax.set_ylim(lag_x[0], lag_x[-1])
    for set_ticks, set_labels, lo, hi in ((ax.set_xticks, ax.set_xticklabels, lag_t[0], lag_t[-1]), (ax.set_yticks, ax.set_yticklabels, lag_x[0], lag_x[-1])):
        set_ticks([lo, 0.5 * (lo + hi), hi])
        set_labels([f"{lo:g}", "", f"{hi:g}"])
    ax.tick_params(labelsize=fontsize - 2, length=tick_length, width=tick_width)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    # Nudge the outer tick labels inward so that they stay within the panel (no labels on a shared axis)
    for labels, align in ((ax.get_xticklabels(), ("left", "right")), (ax.get_yticklabels(), ("bottom", "top"))):
        if labels:
            labels[0].set(**{"ha" if align[0] == "left" else "va": align[0]})
            labels[-1].set(**{"ha" if align[0] == "left" else "va": align[1]})
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    return im
