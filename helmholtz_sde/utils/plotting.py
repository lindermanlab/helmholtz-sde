import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from functools import partial

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

from helmholtz_sde.posterior.encoder import GRUEncoder, NullEncoder
from helmholtz_sde.posterior.posterior import Posterior
from helmholtz_sde.utils.general_helpers import transform_marginals

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union


def time_to_index(t: float, t_max: float, n_steps: int) -> jnp.array:
    t = jnp.clip(t, 0.0, t_max)
    idx = jnp.rint(t / t_max * (n_steps - 1)).astype(jnp.int32)
    return jnp.clip(idx, 0, n_steps - 1)


def plot_marginal_trajectories(xs: jnp.array, t_max: float = 1.0, n_trajs: int = 11, linewidth: float = 2., fontsize: float = 12.):
    """
    Plot sample trajectories vs time along each dimension
    """
    n_trials, n_steps, K = xs.shape
    ts = jnp.linspace(0.0, t_max, n_steps)
    labels = [rf"$\boldsymbol{{x}}({i})$" for i in range(K)]

    ncols = K if K <= 3 else 2
    nrows = int(np.ceil(K / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows), sharex=True)
    axs = np.array(axs).reshape(-1)

    n_trajs = min(int(n_trajs), n_trials)

    for j in range(K):
        ax = axs[j]
        for i in range(n_trajs):
            ax.plot(ts, xs[i, :, j], linewidth=linewidth)
        ax.set_ylabel(labels[j], fontsize=fontsize)
        ax.set_xlabel("t", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize-2)

    # Remove any unused axes
    for ax_j in range(K, len(axs)):
        fig.delaxes(axs[ax_j])

    fig.tight_layout()
    return fig, axs[:K]


def plot_latent_2d_projection(xs: jnp.array, idx1: int = 0, idx2: int = 1, n_trajs: int = 11, linewidth: float = 2.0, fontsize: float = 12.0, equal_aspect: bool = True):
    """
    Plot a 2D projection of latent trajectories: (x[idx1], x[idx2])
    """

    n_trials, n_steps, K = xs.shape
    n_trajs = min(int(n_trajs), int(n_trials))

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 5.0))

    for i in range(n_trajs):
        x = xs[i, :, idx1]
        y = xs[i, :, idx2]

        ax.plot(np.asarray(x), np.asarray(y), linewidth=linewidth)

    ax.set_xlabel(rf"$\boldsymbol{{x}}({idx1})$", fontsize=fontsize)
    ax.set_ylabel(rf"$\boldsymbol{{x}}({idx2})$", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 2)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    ax.autoscale_view()

    fig.tight_layout()
    return fig, ax


def plot_latent_3d_projection(xs: jnp.array, idx1: int = 0, idx2: int = 1, idx3: int = 2, n_trajs: int = 11, linewidth: float = 2.0, fontsize: float = 12.0, equal_aspect: bool = True, xlim: Tuple[float, float] = None, ylim: Tuple[float, float] = None, zlim: Tuple[float, float] = None):
    """
    Plot a 3D projection of latent trajectories: (x[idx1], x[idx2], x[idx3])
    """

    n_trials, n_steps, K = xs.shape
    n_trajs = min(int(n_trajs), int(n_trials))

    fig = plt.figure(figsize=(7.0, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0.0, 0.0, 0.75, 1.0])

    for i in range(n_trajs):
        x = xs[i, :, idx1]
        y = xs[i, :, idx2]
        z = xs[i, :, idx3]

        ax.plot(np.asarray(x), np.asarray(y), np.asarray(z), linewidth=linewidth)

    ax.set_xlabel(rf"$\boldsymbol{{x}}({idx1})$", fontsize=fontsize)
    ax.set_ylabel(rf"$\boldsymbol{{x}}({idx2})$", fontsize=fontsize)
    ax.set_zlabel(rf"$\boldsymbol{{x}}({idx3})$", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 2)
    # Ensure z-axis label shows up
    fig.text(0.99, 0.5, " ", color="white", alpha=0.0)

    # Infer limits from data only if not provided
    sub = xs[:n_trajs, :, jnp.array([idx1, idx2, idx3])]
    mins = jnp.min(sub, axis=(0, 1))
    maxs = jnp.max(sub, axis=(0, 1))

    xmin_data, ymin_data, zmin_data = [float(v) for v in mins]
    xmax_data, ymax_data, zmax_data = [float(v) for v in maxs]

    xmin, xmax = xlim if xlim is not None else (xmin_data, xmax_data)
    ymin, ymax = ylim if ylim is not None else (ymin_data, ymax_data)
    zmin, zmax = zlim if zlim is not None else (zmin_data, zmax_data)

    if equal_aspect:
        cx, cy, cz = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)
        r = 0.5 * max(xmax - xmin, ymax - ymin, zmax - zmin)
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_zlim(cz - r, cz + r)
    else:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

    return fig, ax


def plot_marginal_mean_std(xs: jnp.array, t_max: float = 1.0, num_std: float = 1., fontsize: float = 12.):
    """
    Plot empirical marginal mean and std vs time for each dimension
    """
    n_trials, n_steps, K = xs.shape
    ts = jnp.linspace(0.0, t_max, n_steps)

    emp_mean = xs.mean(axis=0)
    emp_std = xs.std(axis=0, ddof=0)

    labels = [rf"$\boldsymbol{{x}}({i})$" for i in range(K)]

    ncols = K if K <= 3 else 2
    nrows = int(np.ceil(K / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows), sharex=True)
    axs = np.array(axs).reshape(-1)

    for j in range(K):
        ax = axs[j]
        ax.plot(ts, emp_mean[:, j])
        ax.fill_between(ts, emp_mean[:, j] - num_std * emp_std[:, j], emp_mean[:, j] + num_std * emp_std[:, j], alpha=0.25)
        ax.set_title(labels[j], fontsize=fontsize)
        ax.set_xlabel("t", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 2)

    # Remove unused axes
    for ax_j in range(K, len(axs)):
        fig.delaxes(axs[ax_j])

    fig.tight_layout()
    return fig, axs[:K]


def plot_joint_x1_two_times(xs: jnp.array, s: float, t: float, dim: int = 0, t_max: float = 1., title=None, bins: int = 40, fontsize: float = 12.):
    """
    Visualize the joint distribution of (x_i(s), x_i(t)) for s < t and some index i
    """

    s_idx = time_to_index(s, t_max=t_max, n_steps=xs.shape[1])
    t_idx = time_to_index(t, t_max=t_max, n_steps=xs.shape[1])
    Xs = xs[:, s_idx, dim]
    Xt = xs[:, t_idx, dim]

    fig, ax_scatt = plt.subplots(1, 1, figsize=(5, 5))

    # Scatter
    ax_scatt.scatter(Xs, Xt, s=6, alpha=0.25)
    ax_scatt.set_xlabel(rf"$\boldsymbol{{x}}_{{{dim}}}(s)$", fontsize=fontsize)
    ax_scatt.set_ylabel(rf"$\boldsymbol{{x}}_{{{dim}}}(t)$", fontsize=fontsize)

    if title is not None:
        ax_scatt.set_title(title, fontsize=fontsize)

    # Formatting
    lim_low = float(min(Xs.min(), Xt.min()))
    lim_high = float(max(Xs.max(), Xt.max()))
    span = lim_high - lim_low
    pad = 0.05 * span if span > 0 else 1.0
    lim_low -= pad
    lim_high += pad
    hist_range = (lim_low, lim_high)

    ax_scatt.set_xlim(lim_low, lim_high)
    ax_scatt.set_ylim(lim_low, lim_high)
    ax_scatt.set_aspect("equal", adjustable="box")

    divider = make_axes_locatable(ax_scatt)
    ax_histx = divider.append_axes("top", size=1.2, pad=0.15, sharex=ax_scatt)
    ax_histy = divider.append_axes("right", size=1.2, pad=0.15, sharey=ax_scatt)

    # Histograms with matching ranges
    ax_histx.hist(Xs, bins=bins, density=True, range=hist_range)
    ax_histy.hist(Xt, bins=bins, density=True, range=hist_range, orientation="horizontal")

    ax_histx.set_ylabel("density", fontsize=fontsize)
    ax_histy.set_xlabel("density", fontsize=fontsize)

    # Clean ticks
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    fig.tight_layout()
    return fig, (ax_scatt, ax_histx, ax_histy)


def _evaluate_posterior_on_grid(ys_obs: jnp.array, obs_times: jnp.array, post_net: Posterior, encoder: Union[GRUEncoder, NullEncoder], params: Dict[str, Any], process_ctx: Callable[[jnp.array, jnp.array], jnp.array], ts: jnp.array, per_trial_posterior: bool = False) -> Tuple[jnp.array, jnp.array]:
    """
    Posterior means (B, T, K) and covariances (B, T, K, K) of each trial at the times ts (T)
    """
    ctx_seq = vmap(partial(encoder.apply, params["encoder_params"]))(ys_obs)

    def _eval_posterior(post_params_b, ctx_seq_b, obs_times_b, t):
        m, L = post_net.apply(post_params_b, jnp.array([t], dtype=ys_obs.dtype), ctx_seq_b, partial(process_ctx, obs_times_b))
        return m, L @ L.T

    if per_trial_posterior:
        return vmap(lambda pp_b, ctx_b, times_b: vmap(lambda t: _eval_posterior(pp_b, ctx_b, times_b, t))(ts))(params["posterior_params"], ctx_seq, obs_times)
    return vmap(lambda ctx_b, times_b: vmap(lambda t: _eval_posterior(params["posterior_params"], ctx_b, times_b, t))(ts))(ctx_seq, obs_times)


def plot_posterior_marginals(ys_obs: jnp.array, obs_times: jnp.array, post_net: Posterior, encoder: Union[GRUEncoder, NullEncoder], params: Dict[str, Any], process_ctx: Callable[[jnp.array, jnp.array], jnp.array], P: Optional[jnp.array] = None, offset: Optional[jnp.array] = None, posterior_mean_true: Optional[jnp.array] = None, posterior_std_true: Optional[jnp.array] = None, true_latents: Optional[jnp.array] = None, output_params_true: Optional[Dict[str, jnp.array]] = None, t_max: float = 1., n_timesteps: int = 1000, eps_std: float = 1e-8, num_std: float = 1., fontsize: float = 12., dim_mask: Optional[jnp.array] = None, color: Optional[str] = None, xticks: Optional[Sequence[Optional[Sequence[float]]]] = None, yticks: Optional[Sequence[Optional[Sequence[float]]]] = None, xticklabels: Optional[Sequence[Optional[Sequence[str]]]] = None, yticklabels: Optional[Sequence[Optional[Sequence[str]]]] = None, xlims: Optional[Sequence[Optional[Tuple[float, float]]]] = None, ylims: Optional[Sequence[Optional[Tuple[float, float]]]] = None, per_trial_posterior: bool = False):
    """
    Plot the posterior marginals
    """

    B, _, D_obs = ys_obs.shape

    # Evaluate the posterior on a grid
    ts = jnp.linspace(0.0, t_max, n_timesteps + 1)
    ms, Ss = _evaluate_posterior_on_grid(ys_obs, obs_times, post_net, encoder, params, process_ctx, ts, per_trial_posterior=per_trial_posterior) # (B, T, K), (B, T, K, K)
    if P is None:
        P = jnp.eye(ms.shape[-1])
    if offset is None:
        offset = jnp.zeros((ms.shape[-1]))
    ms, Ss = transform_marginals(ms, Ss, P, offset) # transform into the true latent coordinates
    stds = vmap(lambda S_seq: vmap(lambda S: jnp.sqrt(jnp.maximum(jnp.diag(S), eps_std)))(S_seq))(Ss) # (B, T, D)

    if output_params_true is not None: # transform observations into x space
        ys_obs_tilde = vmap(lambda ys: vmap(lambda y: jnp.linalg.solve(output_params_true["C"], y - output_params_true["d"]))(ys))(ys_obs)
    else:
        ys_obs_tilde = ys_obs
    ys_obs = ys_obs_tilde

    K = int(ms.shape[-1])

    if dim_mask is None:
        dim_mask = np.ones((K,), dtype=bool)
    else:
        dim_mask = np.asarray(dim_mask, dtype=bool)
        assert dim_mask.shape == (K,), f"dim_mask must have shape {(K,)}, got {dim_mask.shape}"
        assert dim_mask.sum() == D_obs, f"ys_obs last dim ({D_obs}) must equal dim_mask.sum() ({dim_mask.sum()})"

    labels = [rf"$\boldsymbol{{x}}({i})$" for i in range(K)]
    ncols = K if K <= 3 else 2
    nrows = int(np.ceil(K / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows), sharex=True)
    axs = np.array(axs).reshape(-1)

    ts_np = np.asarray(ts)
    means_np = np.asarray(ms)
    stds_np = np.asarray(stds)
    ys_np = np.asarray(ys_obs)
    ot_np = np.asarray(obs_times)

    if posterior_mean_true is not None:
        posterior_mean_true = np.asarray(posterior_mean_true)

    if posterior_std_true is not None:
        posterior_std_true = np.asarray(posterior_std_true)

    if true_latents is not None:
        true_latents = np.asarray(true_latents)

    obs_dim_counter = 0
    for j in range(K):
        ax = axs[j]
        for b in range(B):
            mu = means_np[b, :, j]
            sd = stds_np[b, :, j]
            (line,) = ax.plot(ts_np, mu, color=color, linewidth=2.0)
            c = line.get_color()

            ax.fill_between(ts_np, mu - num_std * sd, mu + num_std * sd, alpha=0.25, color=c)
            if dim_mask[j]:
                ax.plot(ot_np[b], ys_np[b, :, obs_dim_counter], linestyle="None", marker="x", markersize=6, mew=1.5, color="0.6")

            if posterior_mean_true is not None:
                mu_true = posterior_mean_true[b, :, j]
                ax.plot(ts_np, mu_true, color="#9467bd", linewidth=2.0)
                if posterior_std_true is not None:
                    sd_true = posterior_std_true[b, :, j]
                    ax.plot(ts_np, mu_true + num_std * sd_true, color="black", linestyle=(0, (8, 6)), linewidth=1.5)
                    ax.plot(ts_np, mu_true - num_std * sd_true, color="black", linestyle=(0, (8, 6)), linewidth=1.5)
            if true_latents is not None:
                n_true = true_latents.shape[1]
                ts_true = np.linspace(0.0, float(t_max), n_true)
                z_true = true_latents[b, ::10, j]
                ax.plot(ts_true[::10], z_true, color="0.6", linewidth=1.5, zorder=1)

        if dim_mask[j]:
            obs_dim_counter += 1

        ax.set_ylabel(labels[j], fontsize=fontsize)
        ax.set_xlabel("t", fontsize=fontsize)

        if xticks is not None and xticks[j] is not None:
            ax.set_xticks(xticks[j])
        if yticks is not None and yticks[j] is not None:
            ax.set_yticks(yticks[j])
        if xticklabels is not None and xticklabels[j] is not None:
            ax.set_xticklabels(xticklabels[j])
        if yticklabels is not None and yticklabels[j] is not None:
            ax.set_yticklabels(yticklabels[j])

        ax.tick_params(labelsize=fontsize - 2)

        if xlims is not None and xlims[j] is not None:
            ax.set_xlim(xlims[j])
        if ylims is not None and ylims[j] is not None:
            ax.set_ylim(ylims[j])

    for ax_j in range(K, len(axs)):
        fig.delaxes(axs[ax_j])

    fig.tight_layout()
    return fig, axs[:K], (ms, Ss)


def plot_posterior_samples(xs_post: jnp.array, ys_obs: jnp.array, obs_times: jnp.array, post_net: Posterior, encoder: Union[GRUEncoder, NullEncoder], params: Dict[str, Any], process_ctx: Callable[[jnp.array, jnp.array], jnp.array], P: Optional[jnp.array] = None, offset: Optional[jnp.array] = None, true_latents: Optional[jnp.array] = None, t_max: float = 1., fontsize: float = 12., linewidth: float = 0.8, alpha: float = 0.1, per_trial_posterior: bool = False):
    """
    Plot posterior sample paths xs_post (B, n_samples, T, K) of each trial (rows) along each dimension (columns)
    """
    B, _, T, K = xs_post.shape
    ts = jnp.linspace(0.0, t_max, T)
    ms, _ = _evaluate_posterior_on_grid(ys_obs, obs_times, post_net, encoder, params, process_ctx, ts, per_trial_posterior=per_trial_posterior) # (B, T, K)
    if P is None:
        P = jnp.eye(K)
    if offset is None:
        offset = jnp.zeros((K))
    affine = lambda x: P @ x + offset
    ms_np = np.asarray(vmap(vmap(affine))(ms)) # (B, T, D), in the space of the affine map
    xs_np = np.asarray(vmap(vmap(vmap(affine)))(xs_post)) # (B, n_samples, T, D)
    ts_np, ys_np, ot_np = np.asarray(ts), np.asarray(ys_obs), np.asarray(obs_times)
    D = ms_np.shape[-1]
    labels = [rf"$\boldsymbol{{x}}({j})$" for j in range(D)]

    fig, axs = plt.subplots(B, D, figsize=(5.0 * D, 3.0 * B), sharex=True, squeeze=False)
    for b in range(B):
        color = f"C{b}"
        for j in range(D):
            ax = axs[b, j]
            ax.plot(ts_np, xs_np[b, :, :, j].T, color=color, linewidth=linewidth, alpha=alpha)
            ax.plot(ts_np, ms_np[b, :, j], color=color, linewidth=2.0, zorder=3)
            if true_latents is not None:
                ax.plot(np.linspace(0.0, t_max, true_latents.shape[1]), np.asarray(true_latents)[b, :, j], color="0.6", linewidth=1.5, zorder=2) # same grey as in plot_posterior_marginals
            ax.plot(ot_np[b], ys_np[b, :, j], linestyle="None", marker="x", markersize=6, mew=1.5, color="0.6", zorder=4)
            ax.set_ylabel(labels[j], fontsize=fontsize)
            ax.tick_params(labelsize=fontsize - 2)
        axs[b, 0].set_title(f"Trial {b + 1}", fontsize=fontsize, loc="left")
    for ax in axs[-1]:
        ax.set_xlabel("t", fontsize=fontsize)

    fig.tight_layout()
    return fig, axs


def plot_dynamics_2d(prior: Callable, drift_params: Dict[str, Any], t: float = 0.0, figsize: Tuple[float, float] = (3.0, 3.0), xlim: Tuple[float, float] = (-2.0, 2.0), ylim: Tuple[float, float] = (-2.0, 2.0), n: int = 16, title: Optional[str] = None, fontsize: float = 12.0, xticks: Optional[Union[np.array, list]] = None, yticks: Optional[Union[np.array, list]] = None, xticklabels: Optional[Union[np.array, list]] = None, yticklabels: Optional[Union[np.array, list]] = None, prior_true: Optional[Callable] = None, drift_params_true: Optional[Dict[str, Any]] = None, learned_kwargs: Optional[dict] = None, true_kwargs: Optional[dict] = None):
    """
    Plot a 2D drift field f(x,t) on a spatial grid using a quiver plot
    """

    if learned_kwargs is None:
        learned_kwargs = {}
    if true_kwargs is None:
        true_kwargs = {"alpha": 0.6}

    # Grid
    xs = jnp.linspace(xlim[0], xlim[1], n)
    ys = jnp.linspace(ylim[0], ylim[1], n)
    X, Y = jnp.meshgrid(xs, ys, indexing="xy") # (n, n)

    # Evaluate drift on the grid
    pts = jnp.stack([X.ravel(), Y.ravel()], axis=-1) # (n*n, 2)

    def f_one(x):
        return prior(x, jnp.array(t), drift_params) # (2,)

    U = jax.vmap(f_one)(pts) # (n*n, 2)
    Ux = jnp.reshape(U[:, 0], (n, n))
    Uy = jnp.reshape(U[:, 1], (n, n))

    Xh, Yh = np.asarray(X), np.asarray(Y)
    Uxh, Uyh = np.asarray(Ux), np.asarray(Uy)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.quiver(Xh, Yh, Uxh, Uyh, angles="xy", scale_units="xy", **learned_kwargs)

    # Optionally overlay ground-truth field
    if prior_true is not None:
        def f_true_one(x):
            return prior_true(x, jnp.array(t), drift_params_true) # (2,)

        U_true = jax.vmap(f_true_one)(pts) # (n*n, 2)
        Ux_true = jnp.reshape(U_true[:, 0], (n, n))
        Uy_true = jnp.reshape(U_true[:, 1], (n, n))

        Uxh_true = np.asarray(Ux_true)
        Uyh_true = np.asarray(Uy_true)

        ax.quiver(Xh, Yh, Uxh_true, Uyh_true, angles="xy", scale_units="xy", **true_kwargs)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"$x_1$", fontsize=fontsize)
    ax.set_ylabel(r"$x_2$", fontsize=fontsize)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    else:
        ax.set_title(rf"Learned drift at $t={t}$", fontsize=fontsize)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    ax.tick_params(labelsize=fontsize - 2)
    fig.tight_layout()
    return fig, ax


def _concise_tick(value, _pos):
    """
    Concise y-tick formatter that keeps labels narrow
    """
    if value == 0:
        return "0"
    magnitude = abs(value)
    if 1e-2 <= magnitude < 1e4:
        return f"{value:.3g}"
    return f"{value:.0e}"


def plot_losses(metrics: Dict[str, Any], keys: Sequence[str] = ("loss", "kl", "rec", "prior"), titles: Optional[Sequence[str]] = None, figsize: Optional[Tuple[float, float]] = None, fontsize: float = 12.0, linewidth: float = 0.8, yscale: str = "symlog", xticks: Optional[Sequence] = None, yticks: Optional[Sequence] = None, xticklabels: Optional[Sequence] = None, yticklabels: Optional[Sequence] = None):
    """
    Plot training loss curves, one panel per quantity
    By default shows the total loss, the KL term, the reconstruction term, and the initial KL (KL0)
    metrics is the dict returned by train, holding a per-iteration array for each key
    """

    default_titles = {"loss": "Total loss", "kl": "KL", "rec": "Reconstruction", "prior": "KL0 (initial KL)"}
    if titles is None:
        titles = [default_titles.get(k, k) for k in keys]
    n = len(keys)
    if figsize is None:
        figsize = (4.0 * n, 3.5)

    fig, axs = plt.subplots(1, n, figsize=figsize, squeeze=False, constrained_layout=True)
    axs = axs[0]

    # Broadcast a single tick/label spec to every panel, or accept one spec per panel
    def _per_panel(val):
        if val is None:
            return [None] * n
        if isinstance(val[0], (list, np.ndarray)):
            return list(val)
        return [val] * n
    xticks, yticks = _per_panel(xticks), _per_panel(yticks)
    xticklabels, yticklabels = _per_panel(xticklabels), _per_panel(yticklabels)

    for j, (ax, key, title) in enumerate(zip(axs, keys, titles)):
        ax.plot(np.asarray(metrics[key]), linewidth=linewidth)
        ax.set_yscale(yscale)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_concise_tick))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel("iteration", fontsize=fontsize)

        if xticks[j] is not None:
            ax.set_xticks(xticks[j])
        if yticks[j] is not None:
            ax.set_yticks(yticks[j])
        if xticklabels[j] is not None:
            ax.set_xticklabels(xticklabels[j])
        if yticklabels[j] is not None:
            ax.set_yticklabels(yticklabels[j])

        ax.tick_params(labelsize=fontsize - 2)

    return fig, axs
