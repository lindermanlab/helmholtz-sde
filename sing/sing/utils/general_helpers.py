"""
Contains general helper functions that are not specific to the SING algorithm 
e.g., for data cleaning, analyzing results from SING
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
from jax import vmap, lax

from functools import partial

import flax.linen as nn # import flax for the MLP

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from sing.kernels import Kernel

import optax
import numpy as np

# --------------------- Helpers for sampling and binning data -------------------
def discretize_sde_on_grid(t_grid: jnp.array, As: jnp.array, bs: jnp.array, L: jnp.array) -> Tuple[jnp.array, jnp.array, jnp.array]:
    """
    Discretizes a linear SDE from evaluations of the drift term on the specified grid
    using the Euler-Maruyama scheme

    Params
    ----------
    t_grid: a shape (T) array
    As: a shape (T - 1, D, D) array, the matrix part of the drift evaluated on the grid \tau
    bs: a shape (T - 1, D) array, the vector part of the drift evaluated on the grid \tau
    L: a shape (D, D) array, the diffusion coefficient L dW(t) 
    """
    d = As.shape[-1]

    del_t = t_grid[1:] - t_grid[:-1] # (T - 1)

    As_discretized = jnp.eye(d) + (del_t[:, None, None] * As)
    bs_discretized = (del_t[:, None] * bs)
    Q_discretized = del_t[:, None, None] * (L @ L.T)[None,:,:]

    return As_discretized, bs_discretized, Q_discretized

def bin_regularly_sampled_data(dt: float, ys: jnp.array, bin_size: float) -> Tuple[jnp.array, jnp.array]:
    """
    Bin regularly-sampled observations into **smaller** time bins.

    Params
    ----------
    dt: timestep used in EM algorithm
    ys: a shape (n_trials, n_bins, N) array, observed data
    bin_size: amount of time per time bin in ys (must be same unit as dt), 
    NOTE: bin_size represents the current time per time bin
    """
    n_trials, n_bins, D = ys.shape
    trial_duration = n_bins * bin_size 
    t_obs = np.arange(n_bins) * bin_size # time-stamps of observations
    t_mask, _ = np.histogram(t_obs, int(trial_duration / dt), (0, trial_duration)) # t_mask is the same across trials
    t_mask = t_mask.astype(bool) 
    ys_binned = np.zeros((n_trials, len(t_mask), D))
    ys_binned[:,t_mask.astype(bool),:] = ys
    t_mask = t_mask[None].repeat(n_trials, axis=0) # repeat across trial dimension
    
    return jnp.array(ys_binned), jnp.array(t_mask)
    
def bin_sparse_data(ys: jnp.array, t_obs: jnp.array, t_max: float, dt: float) -> Tuple[jnp.array, jnp.array]:
    """
    Bin sparsely sampled data (at potentially irregular times) into discrete time bins

    Params:
    -------------
    ys: a shape (n_trials, n_samps, N) array, the sparse observations
    t_obs: a shape (n_trials, n_samps) array, the timestamps of observations
    t_max: scalar, duration of trials
    dt: scalar, bin size

    Returns:
    -------------
    ys_binned: a shape (n_trials, T, N) array, binned observations
    t_mask: a shape (n_trials, T) array, mask for observed timestamps
    """

    T = int(t_max / dt)
    n_trials, n_samps, D = ys.shape
    all_t_mask = []
    all_ys_binned = []

    for i in range(n_trials):
        hist, bins = np.histogram(t_obs[i], T, (0, t_max)) # (T) containing counts of obs in each time bin
        t_idx = np.nonzero(hist)[0] # containing time bin indices with >= 1 obs
        ys_binned = np.zeros((T, D))
        for j, idx in enumerate(t_idx):
            if j < len(t_idx) - 1:
                y_inds_in_bin = np.nonzero((bins[idx] <= t_obs[i]) & (t_obs[i] < bins[idx+1]))[0] # get indices of n_samps where obs are in this bin
                ys_binned[idx] = ys[i, y_inds_in_bin].mean(0) # taken mean of those obs
            else:
                y_inds_in_bin = np.nonzero((bins[idx] <= t_obs[i]) & (t_obs[i] <= bins[idx+1]))[0] # get indices of n_samps where obs are in this bin
                ys_binned[idx] = ys[i, y_inds_in_bin].mean(0) # taken mean of those obs

        all_t_mask.append(hist != 0)
        all_ys_binned.append(ys_binned)

    all_t_mask = np.stack(all_t_mask)
    all_ys_binned = np.stack(all_ys_binned)
    return jnp.array(all_ys_binned), jnp.array(all_t_mask)

# --------------------- Helpers for analyzing SING inferred latents -------------------
def get_transformation_for_latents(C: jnp.array, d: jnp.array, C_hat: jnp.array, d_hat: jnp.array) -> Tuple[jnp.array, jnp.array]:
    """
    Computes a linear transformation matrix 
    Px + offset
    that maps learned latents to true latents (or another latent space)

    Params:
    ------------
    C: a shape (N, D) array, the true linear mapping parameter
    d: a shape (N) array, the true offset parameter
    C_hat: a shape (N, D) array, the learned linear mapping parameter
    d_hat: a shape (N) array, the learned offset parameter

    Returns:
    ------------
    A mapping Px + offset from learned -> true latents
    P: a shape (D, D) array, the linear part of the map
    offset: a shale (D) array, the additive part of the map
    """    
    U, S, Vt = jnp.linalg.svd(C, full_matrices=False)
    MP_inv = Vt.T @ jnp.diag(1./S) @ U.T
    P = MP_inv @ C_hat
    offset = MP_inv @ (d_hat - d)
    return P, offset

def get_learned_partition(partition_fn, kernel_params: Dict[str, jnp.array], Xs: jnp.array) -> jnp.array:
    """
    For the SSL (smoothly switching linear) kernel, compute value of pi(x) at each point in Xs

    Params:
    ------------
    partition_fn: construct_partition function from SSL kernel class
    kernel_params: dictionary containing the params of the SSL kernel
    Xs: a shape (n_points, D) array, the batch of points on which to evaluate pi

    Returns:
    ------------
    learned_pis: a shape (n_points, num_states) array, pi evaluated at Xs
    """
    learned_pis = vmap(partition_fn, (0, None, None))(Xs, kernel_params['W'], kernel_params['log_tau'])
    return learned_pis
    
def get_most_likely_state(partition_fn, kernel_params: Dict[str, jnp.array], Xs: jnp.array) -> jnp.array:
    """For SSL kernel, compute most likely state at each point in Xs."""
    learned_pis = get_learned_partition(partition_fn, kernel_params, Xs)
    most_likely_states = jnp.argmax(learned_pis, 1)
    return most_likely_states

def compute_latents_mse(xs: jnp.array, ms: jnp.array, Ss: Optional[jnp.array] = None) -> float:
    """
    Computes the MSE between the latents under the variational posterior and the true latents

    Params:
    ------------
    xs: a shape (T, D) array, the true latents
    latents_mean: a shape (T, D) array, the (univariate) marginal means of the latents under the variational posterior q
    latents_cov, optional: a shape (T, D, D) array, the marginal covariance of the latents
    """
    def _compute_mse_single(true, m, S):
        """
        Compute E[||x-true||^2] where x ~ N(m, S).
        - true: a shape (D) array
        - m: a shape (D) array
        - S: a shape (D, D) array
        """
        return jnp.trace(S) + ((m - true)**2).sum()

    n_timesteps, latent_dim = xs.shape
    if Ss is None:
        Ss = jnp.zeros(n_timesteps, latent_dim, latent_dim)

    # vmap over timesteps
    return vmap(_compute_mse_single)(xs, ms, Ss).mean()

# --------------------- Other helpers -------------------
def make_gram(kernel_fn: Kernel, kernel_params: Dict[str, Any], Xs: jnp.array, Xps: jnp.array, jitter=1e-8) -> jnp.array:
    """
    Compute gram matrix between inputs Xs and Xps.

    Params:
    ---------------
    kernel_fn: function from Kernel class
    kernel_params: dictionary containing the kernel parameters 
    Xs: a shape (n_points, K) array, the first batch of input points
    Xps: a shape (n_points_2, K) array, the second batch of input points
    jitter: optional jitter to add to gram matrix, should be None if Xs != Xps

    Returns:
    ---------------
    K: a shape (n_points, n_points_2) array, the gram matrix
    """
    K = vmap(vmap(partial(kernel_fn, kernel_params=kernel_params), (None, 0)), (0, None))(Xs, Xps)
    if jitter is not None:
        K += jitter * jnp.eye(len(Xs))
    return K

def sgd(key: jr.PRNGKey, loss_fn: Callable[[jr.PRNGKey, Dict[str, Any]], float], params: Dict[str, Any], n_iters: int, learning_rate: float, learning_rate_final: Optional[float] = None, max_grad_norm: float = 5.0) -> Tuple[Dict[str, Any], float]:
    """
    Performs SGD with Adam on a specified loss function with respect to params

    Params:
    ------------
    key: random key for sampling; the loss function can depend on this random key i.e. L(key, x) -> R_+
    loss_fn: loss function to be optimizing, a function of params
    params: dictionary containing the parameters with respect to which the loss function is optimized
    n_iters: the number of SGD iterations
    learning_rate: the learning rate for Adam
    learning_rate_final: optional final learning rate; if provided, uses exponential decay from learning_rate to learning_rate_final over n_iters steps
    max_grad_norm: maximum global norm for gradient clipping

    Returns:
    ------------
    final_params: the parameters from the final step of SGD
    final_val: final value of the loss function
    """
    if learning_rate_final is not None:
        decay_rate = learning_rate_final / learning_rate
        schedule = lambda count: learning_rate * (decay_rate ** (count / n_iters))
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(schedule),
        )
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate),
        )
    opt_state = optimizer.init(params)

    def _step(carry, arg):
        params_prev, opt_state_prev = carry
        key = arg

        loss, grads = jax.value_and_grad(loss_fn, argnums=1)(key, params_prev)
        updates, opt_state = optimizer.update(grads, opt_state_prev, params_prev)
        params = optax.apply_updates(params_prev, updates)
        return (params, opt_state), -loss # returning elbo

    initial_carry = (params, opt_state)
    args = jr.split(key, n_iters)
    (final_params, _), all_elbos = lax.scan(_step, initial_carry, args, length=n_iters)
    final_val = all_elbos[-1]

    return final_params, final_val

# --------------------- Helpers for neural-SDEs -------------------
class MLP(nn.Module):
    """
    A simple flax implementation of a multilayer perceptron (MLP) with ReLU activations
    """
    features: Sequence[int]  # list of hidden‐layer sizes
    latent_dim: int # size of the output layer

    @nn.compact
    def __call__(self, x: jnp.array, t: float) -> jnp.array:
        xt = x # for now, neural network drift is time homogeneous
        h = xt
        for feat in self.features:
            h = nn.relu(nn.Dense(feat)(h))
        return nn.Dense(self.latent_dim)(h)