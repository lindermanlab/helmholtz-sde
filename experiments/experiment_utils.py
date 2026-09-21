"""
Code shared by the experiments
"""

from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from helmholtz_sde.sde import SDE
from helmholtz_sde.likelihood import Likelihood
from helmholtz_sde.posterior.posterior import Posterior
from helmholtz_sde.posterior.encoder import NullEncoder, ForwardGRUEncoder
from helmholtz_sde.posterior.drift import apply_inference_net_time_derivs, get_reference_drift_fn
from helmholtz_sde.train import prior_loss
from helmholtz_sde.utils.general_helpers import compute_kl, reconstruction_loss_exact
from helmholtz_sde.helmholtz.correction import HelmCorrection, compute_kl_helmholtz_approx
from helmholtz_sde.helmholtz.taylor import TaylorCorrection
from helmholtz_sde.helmholtz.subspace import LeastSquaresCorrection


# --------------------- Helmholtz correction ---------------------
def build_correction(div_free: str, ell: int, kappa: int, n_mc: int, n_nodes: Optional[int] = None) -> Optional[HelmCorrection]:
    """
    Helmholtz correction used during training, from the command-line flags: none (SDE Matching), taylor or least_squares
    """
    if div_free == "none":
        return None
    if div_free == "taylor":
        return TaylorCorrection(ell=ell)
    return LeastSquaresCorrection(ell=ell, kappa=kappa, n_mc=n_mc, n_nodes=n_nodes)


def eval_correction(div_free: Optional[HelmCorrection], n_nodes: int) -> Optional[HelmCorrection]:
    """
    Correction used for evaluating the learned posterior (not training): a least-squares projection with Gauss-Hermite
    quadrature, deterministic given the marginals
    """
    if div_free is None:
        return None
    return LeastSquaresCorrection(ell=div_free.ell, kappa=0, n_nodes=n_nodes, jitter=div_free.jitter)


# --------------------- ELBO evaluation ---------------------
def eval_inference_network(key: jr.PRNGKey, ctx_seq: jnp.array, process_ctx: Callable[[jnp.array, jnp.array], jnp.array], post_net: Posterior, params: Dict[str, Any], prior: SDE, likelihood: Likelihood, ys: jnp.array, obs_times: jnp.array, t_max: float = 1.0, n_z0: int = 128, grid_size: int = 128, jitter: float = 1e-8, gauge: str = "sqrt", div_free: Optional[HelmCorrection] = None) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    """
    Negative ELBO of one trial and its KL, reconstruction and prior terms, with the KL integral evaluated on a regular
    grid of grid_size intervals with n_z0 Monte Carlo states per grid point and the reconstruction term at every observation
    NOTE: the ELBO is invariant under invertible transformations of the latent space, so learned output parameters need no alignment
    """
    key_kl, key_ell = jr.split(key, 2)
    ts = jnp.linspace(0.0, t_max, grid_size + 1)
    dt = ts[1] - ts[0]

    def _kl_rate(t: jnp.array, key_x: jr.PRNGKey) -> jnp.array:
        t_arr = jnp.array([t])
        mt, Rt, dmt, dRt = apply_inference_net_time_derivs(post_net, params["posterior_params"], t_arr, ctx_seq, process_ctx)
        drift_fn = get_reference_drift_fn(gauge)
        fp = lambda x: prior(x, t, params["sde_params"])
        if div_free is not None:
            Gt = prior.diffusion(jnp.zeros((prior.K)), t, params["sde_params"])
            fq = drift_fn(mt, Rt, dmt, dRt, Gt)
            return compute_kl_helmholtz_approx(m=mt, S=Rt @ Rt.T, fp=fp, fq=fq, Gt=Gt, key=key_x, n_z0=n_z0, correction=div_free)
        return compute_kl(t=t, m=mt, S=Rt @ Rt.T, fp=fp, prior=prior, params=params, drift_fn=partial(drift_fn, mt, Rt, dmt, dRt), key=key_x, n_z0=n_z0, jitter=jitter)

    kl_term = dt * jnp.sum(vmap(_kl_rate)(ts[:-1], jr.split(key_kl, grid_size)))
    rec_term = reconstruction_loss_exact(key_ell, ctx_seq, process_ctx, params, post_net, ys, obs_times, likelihood)
    prior_term = prior_loss(ctx_seq, process_ctx, params, post_net, jitter)
    nelbo = kl_term + rec_term + prior_term
    inf = jnp.array(jnp.inf, dtype=nelbo.dtype)
    return tuple(jnp.where(jnp.isnan(nelbo), inf, x) for x in (nelbo, kl_term, rec_term, prior_term)) # a NaN in any term sets all four to inf


def batched_mean_eval(fn: Callable[[int, int], Tuple[jnp.array, ...]], B: int, batch_size: int = 128) -> Tuple[jnp.array, ...]:
    sums = None
    for i in range(0, B, batch_size):
        j = min(i + batch_size, B)
        batch = [(j - i) * x for x in fn(i, j)]
        sums = batch if sums is None else [s + x for s, x in zip(sums, batch)]
    return tuple(s / B for s in sums)


def batched_eval_elbo(eval_key: jr.PRNGKey, ys_obs: jnp.array, obs_times: jnp.array, encoder: Union[NullEncoder, ForwardGRUEncoder], post_net: Posterior, params: Dict[str, Any], prior: SDE, likelihood: Likelihood, process_ctx: Callable, t_max: float, gauge: str, div_free: Optional[HelmCorrection] = None, batch_size: int = 128, per_trial_posterior: bool = False) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    """
    Negative ELBO and its KL, reconstruction and prior terms averaged over trials, evaluated in batches of batch_size trials
    """
    def _eval_trial(key: jr.PRNGKey, ys: jnp.array, obs_t: jnp.array, post_params: Dict[str, Any]) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
        params_trial = {**params, "posterior_params": post_params}
        ctx_seq = encoder.apply(params_trial["encoder_params"], ys)
        return eval_inference_network(key, ctx_seq, partial(process_ctx, obs_t), post_net, params_trial, prior, likelihood, ys, obs_t, t_max=t_max, gauge=gauge, div_free=div_free)

    def _eval_batch(i: int, j: int) -> Tuple[jnp.array, ...]:
        keys = jr.split(jr.fold_in(eval_key, i), j - i)
        post_params = jax.tree.map(lambda x: x[i:j], params["posterior_params"]) if per_trial_posterior else params["posterior_params"]
        out = vmap(_eval_trial, in_axes=(0, 0, 0, 0 if per_trial_posterior else None))(keys, ys_obs[i:j], obs_times[i:j], post_params)
        return tuple(jnp.mean(x) for x in out)

    return batched_mean_eval(_eval_batch, ys_obs.shape[0], batch_size=batch_size)
