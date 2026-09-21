"""
Helpers for the Ornstein-Uhlenbeck (OU) spiral experiments
"""

import contextlib
import os
from functools import partial
from typing import Dict, Tuple

import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from helmholtz_sde.utils.plotting import time_to_index

from sing.likelihoods import Gaussian as SINGGaussian
from sing.sde import LinearSDE as SINGLinearSDE
from sing.sing import fit_variational_em


# --------------------- SING smoother ---------------------
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
        yield


def run_sing_smoother(ys: jnp.array, obs_times: jnp.array, t_grid: jnp.array, sde_params: Dict[str, jnp.array], init_params: Dict[str, jnp.array], output_params: Dict[str, jnp.array]) -> Tuple[jnp.array, jnp.array, jnp.array]:
    """
    Exact posterior of the linear-Gaussian model on the grid t_grid from one E-step of SING (Hu et al., 2025): marginal
    means (B, T, K), covariances (B, T, K, K) and the cross-covariances Cov(x(t_{k+1}), x(t_k)) (B, T - 1, K, K)
    """
    B, _, D = ys.shape
    K = sde_params["A"].shape[0]
    n_timesteps = t_grid.shape[0] - 1
    idx_obs = vmap(vmap(partial(time_to_index, t_max=t_grid[-1], n_steps=n_timesteps + 1)))(obs_times) # (B, n_obs)

    # Observations on the dense grid, with a mask of the observed times
    b_idx = jnp.arange(B)[:, None]
    t_mask = jnp.zeros((B, n_timesteps + 1), dtype=bool).at[b_idx, idx_obs].set(True)
    ys_dense = jnp.zeros((B, n_timesteps + 1, D), dtype=ys.dtype).at[b_idx, idx_obs, :].set(ys)
    likelihood = SINGGaussian(ys_dense, t_mask)
    init_params_batch = {"mu0": jnp.broadcast_to(init_params["mu0"], (B, K)), "V0": jnp.broadcast_to(init_params["V0"], (B, K, K))}

    with suppress_output(): # NOTE: the random key is irrelevant since one E-step is exact for a linear SDE
        marginal_params, *_ = fit_variational_em(jr.PRNGKey(0), SINGLinearSDE(latent_dim=K), likelihood, t_grid, sde_params, init_params_batch, output_params, batch_size=None, rho_sched=jnp.ones(1), n_iters=1, n_iters_e=1, perform_m_step=False, learn_output_params=False)
    return marginal_params["m"], marginal_params["S"], marginal_params["SS"]
