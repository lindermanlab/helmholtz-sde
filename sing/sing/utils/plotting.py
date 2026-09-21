import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from sing.utils.general_helpers import get_most_likely_state
from sing.kernels import Kernel

import matplotlib.pyplot as plt
from matplotlib import ticker, cm, colors

from typing import Callable, Optional

def transform_vector_field(f: Callable[jnp.array, jnp.array], dim: int, P: Optional[jnp.array] = None, offset: Optional[jnp.array] = None):
    """
    Transforms a vector field according to x' = Px + offset 

    Params:
    ------------
    f: the vector field, has signature f(x) -> R^D for x in R^D
    NOTE: f is a vector field defined on x, not x'
    dim: the dimension D of the vector field
    P: a shape (D, D) array, the linear part of the transformation
    offset: a shape (D) array, the additive part of the transformation 

    Returns:
    ------------
    f_trans: the vector field f defined on x'
    """

    if P is None:
        P = jnp.eye(dim)
    if offset is None:
        offset = jnp.zeros(dim)

    def f_trans(x):
        x_inf = jnp.linalg.solve(P, x - offset)
        return P @ f(x_inf)
    return f_trans

def plot_latents_over_time(t_grid: jnp.array, latents_mean: jnp.array, latents_cov: Optional[jnp.array] = None, ax = None, figsize = (4, 3), fontsize = 12, color = 'red', alpha = 0.2):
    """
    Plots latent trajectories over time

    Params:
    ------------
    t_grid: a shape (T) array, the grid on which to plot trajectories
    latents_mean: a shape (n_trials, T, D) array, the latent trajectories (or their means)
    latents_cov: a shape (n_trials, T, D, D) array, the marginal covariance of the latent trajectorie
    ax: the matplotlib axis on which to plot the latent trajectories; if None, instantiates a new axis

    Returns:
    ------------
    ax: the axis on which the latents are plotted
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    D = latents_mean.shape[-1]

    for i in range(D):
        ax.plot(t_grid, latents_mean[:,i], color=color)

    if latents_cov is not None:
        for i in range(D):
            ax.fill_between(t_grid, latents_mean[:,i]+2*jnp.sqrt(latents_cov[:,i,i]),
                    latents_mean[:,i]-2*jnp.sqrt(latents_cov[:,i,i]), facecolor=color, alpha=alpha)

    ax.set_xlabel("t", fontsize=fontsize)
    ax.set_ylabel(r"$x(t)$", fontsize=fontsize)
    return ax

def plot_dynamics_1d(dynamics_fn: Callable[jnp.array, jnp.array], dynamics_var_fn: Optional[Callable[jnp.array, jnp.array]] = None, xlim = (-2, 2), n_xpts = 20, ax = None, figsize = (4, 3), fontsize = 12, color = 'blue', alpha = 0.2):
    """
    Plots a 1D dynamics function. 

    Params:
    ------------
    dynamics_fn: the dynamics function to be plotted, has signature dynamics_fn(x) -> R
    dynamics_var_fn: the variance of the dynamics, has signature dynamics_var_fn(x) -> R

    NOTE: dynamics_var_fn is only relevant for SparseGP instances
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    input_locs = jnp.linspace(*xlim, n_xpts)[:,None]
    fs = vmap(dynamics_fn)(input_locs)

    ax.plot(input_locs[:,0], fs[:,0], color=color)
    
    if dynamics_var_fn is not None:
        fs_var = vmap(dynamics_var_fn)(input_locs)
        ax.fill_between(input_locs[:,0], fs[:,0] + 2 * jnp.sqrt(fs_var), fs[:,0] - 2 * jnp.sqrt(fs_var), facecolor=color, alpha=alpha)

    ax.set_xlabel("x", fontsize=fontsize)
    ax.set_ylabel(r"$f(x)$", fontsize=fontsize)
    return ax

def plot_dynamics_and_latents_2d(dynamics_fn: Callable[jnp.array, jnp.array], latents: Optional[jnp.array]=None, xlim=(-2, 2), ylim=(-2, 2), n_xpts=20, n_ypts=20, ax=None, figsize=(3, 3), fontsize=12, latents_alpha=1., dynamics_alpha=1.):
    """
    Plots a 2D dynamics function. 

    Params:
    ------------
    dynamics_fn: the dynamics function to be plotted, has signature dynamics_fn(x) -> R^2
    latents: a shape (n_trials, T, 2) array of (mean) latent trajectories. If None, only dynamics will be plotted.
    """
    assert latents is None or latents.shape[-1] == 2

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    x, y = jnp.meshgrid(
        jnp.linspace(*xlim, n_xpts),
        jnp.linspace(*ylim, n_ypts)
    )
    input_locs = jnp.column_stack([x.ravel(), y.ravel()])
    fs = vmap(dynamics_fn)(input_locs)

    if latents is not None:
        for latent in latents:
            ax.plot(latent[:,0], latent[:,1], alpha=latents_alpha)
        
    ax.quiver(input_locs[:,0], input_locs[:,1], fs[:,0], fs[:,1], angles='xy', color='black', alpha=dynamics_alpha)
    ax.set_xlabel(r"$x_1$", fontsize=fontsize)
    ax.set_ylabel(r"$x_2$", fontsize=fontsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return ax

def plot_dynamics_variance_2d(dynamics_var_fn: Callable[jnp.array, jnp.array], log_levels: jnp.array, xlim=(-2, 2), ylim=(-2, 2), n_xpts=40, n_ypts=40, fig=None, ax=None, figsize=(3, 3), fontsize=12, cmap='Purples'):
    """
    Plots the variance of 2D dynamics as a filled contour plot.

    Params:
    ------------
    dynamics_var_fn: the variance of the dynamics, has signature dynamics_var_fn(x) -> R
    log_levels: log levels of the contour plot
    
    NOTE: only relevant for SparseGP instances
    NOTE: the variance of 2D dynamics will be a 2 x 2 matrix, so one first needs to reduce to a scalar quantity
    ex. when the components of the dynamics are uncorrelated, take the determinant of the variance
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    x, y = jnp.meshgrid(
        jnp.linspace(*xlim, n_xpts),
        jnp.linspace(*ylim, n_ypts)
    )
    input_locs = jnp.column_stack([x.ravel(), y.ravel()])
    fs_var = vmap(dynamics_var_fn)(input_locs)
    fs_var_grid = fs_var.reshape(n_ypts, n_xpts)

    contour = ax.contourf(x, y, fs_var_grid, levels=log_levels, alpha=0.5, locator=ticker.LogLocator(), cmap=cmap)
    ax.set_xlabel(r"$x_1$", fontsize=fontsize)
    ax.set_ylabel(r"$x_2$", fontsize=fontsize)
    return fig, ax, contour

def plot_slow_points_2d(dynamics_mean_fn: Callable[jnp.array, jnp.array], dynamics_var_fn: Callable[jnp.array, jnp.array], xlim=(-2, 2), ylim=(-2, 2), n_xpts=20, n_ypts=20, eps=0.1, alpha=0.6, fig=None, ax=None, figsize=(5, 5)):
    """
    Identifies slow points in 2D dynamics.

    NOTE: only relevant for SparseGP instances
    """
    def _compute_prob_slow_point(mean, var, eps):
        normal = tfd.Normal(mean, jnp.sqrt(var))
        return normal.cdf(eps) - normal.cdf(-eps)
    
    x, y = jnp.meshgrid(
        jnp.linspace(*xlim, n_xpts),
        jnp.linspace(*ylim, n_ypts)
    )
    input_locs = jnp.column_stack([x.ravel(), y.ravel()])
    f_mean = vmap(dynamics_mean_fn)(input_locs)
    f_var = vmap(dynamics_var_fn)(input_locs)

    probs_dim1 = vmap(partial(_compute_prob_slow_point, eps=eps))(f_mean[:,0], f_var)
    probs_dim2 = vmap(partial(_compute_prob_slow_point, eps=eps))(f_mean[:,1], f_var)
    slow_point_probs = probs_dim1 * probs_dim2 # assuming independent dimensions
    slow_point_probs_grid = slow_point_probs.reshape(n_ypts, n_xpts)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(slow_point_probs_grid, vmin=0, vmax=1, alpha=alpha, interpolation='bilinear', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], cmap='Purples', origin='lower', aspect='auto') 
    return ax, im
    
def plot_most_likely_states(kernel: Kernel, dynamics_fn: Callable[jnp.array, jnp.array], kernel_params: dict[jnp.array, jnp.array], color_names: list, xlim=(-2, 2), ylim=(-2, 2), n_xpts=20, n_ypts=20, scale=50, alpha=0.7, ax=None, figsize=(3, 3), fontsize=12):
    """
    For the SSL kernel: plots the most likely state in the latent space according to the learned partition function
    """ 
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    cmap = colors.ListedColormap(color_names)
    
    x, y = jnp.meshgrid(
        jnp.linspace(*xlim, n_xpts),
        jnp.linspace(*ylim, n_ypts)
    )
    input_locs = jnp.column_stack([x.ravel(), y.ravel()])
    most_likely_states = get_most_likely_state(kernel.construct_partition, kernel_params, input_locs)
    distinct_states = jnp.unique(most_likely_states)
    f_means = vmap(dynamics_fn)(input_locs)
    
    for state in distinct_states:
        state_inds = (most_likely_states == state)
        ax.quiver(input_locs[state_inds,0], input_locs[state_inds,1], f_means[state_inds,0], f_means[state_inds,1], color=cmap(most_likely_states[state_inds]), angles='xy', scale=scale, alpha=alpha)
    
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return ax
    
def plot_spikes(spikes: jnp.array, t_grid: jnp.array, t_max: float, figsize=(4, 3), ax=None):
    """
    Plots (binary) spike times for a given trial.

    Params:
    ------------------
    spikes: a shape (n_timesteps, D) array, spike counts
    t_grid: a shape (n_timesteps) array, time points corresponding to spikes
    t_max: maximum time to plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    num_neurons = spikes.shape[-1]
    for neuron_idx in range(num_neurons):
        t_obs = t_grid[spikes[:,neuron_idx] > 0]
        ax.vlines(t_obs, neuron_idx - 0.5, neuron_idx + 0.5, color='black')
    ax.set_xlim(0, t_max)
    ax.set_ylim(-0.5, num_neurons - 0.5)
    return ax