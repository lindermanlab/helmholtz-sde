"""
Implements the training loop for the simulation-free VI in the latent SDE model
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

import optax
from functools import partial

import tqdm

from typing import Any, Callable, Dict, Optional, Tuple, Union

from helmholtz_sde.helmholtz.correction import HelmCorrection, compute_kl_helmholtz_approx
from helmholtz_sde.helmholtz.subspace import DEFAULT_CORRECTION
from helmholtz_sde.posterior.encoder import GRUEncoder, NullEncoder
from helmholtz_sde.posterior.posterior import Posterior
from helmholtz_sde.posterior.drift import apply_inference_net_time_derivs, get_reference_drift_fn
from helmholtz_sde.likelihood import Likelihood
from helmholtz_sde.sde import SDE
from helmholtz_sde.utils.general_helpers import compute_kl, gaussian_kl
from helmholtz_sde.data import DataProcessor, process_data_identity


def kl_loss(key: jr.PRNGKey, ctx_seq: jnp.array, process_ctx: Callable[[jnp.array, jnp.array], jnp.array], params: Dict[str, jnp.array], post_net: Posterior, t0: float, t1: float, prior: SDE, gauge: str = "sqrt", div_free: Optional[HelmCorrection] = None, jitter: float = 1e-8) -> float:
    """
    KL divergence between approximate posterior and prior KL(q(•|x(0))||p(•|x(0)))
    """

    key_x, key_t = jr.split(key, 2)

    dt = jnp.maximum(t1 - t0, 0.0)
    u = jr.uniform(key_t, shape=(), minval=0.0, maxval=1.0)
    t = t0 + dt * u
    t = jnp.array([t], dtype=ctx_seq.dtype)

    # Compute posterior
    mt, Rt, dmt, dRt = apply_inference_net_time_derivs(post_net, params["posterior_params"], t, ctx_seq, process_ctx)
    drift_fn = get_reference_drift_fn(gauge)

    # Compute estimate to KL
    fp = lambda x: prior(x, t, params["sde_params"])

    if div_free is not None:
        # NOTE: the Helmholtz correction assumes the diffusion coefficient is state-independent
        Gt = prior.diffusion(jnp.zeros((prior.K)), t, params["sde_params"])
        fq = drift_fn(mt, Rt, dmt, dRt, Gt)

        kl_est = compute_kl_helmholtz_approx(
            m=mt,
            S=Rt @ Rt.T,
            fp=fp,
            fq=fq,
            Gt=Gt,
            key=key_x,
            n_z0=1,
            correction=div_free,
        )
    else: # no Helmholtz correction
        kl_est = compute_kl(
            t=t,
            m=mt,
            S=Rt @ Rt.T,
            fp=fp,
            prior=prior,
            params=params,
            drift_fn=partial(drift_fn, mt, Rt, dmt, dRt),
            key=key_x,
            n_z0=1,
            jitter=jitter,
        )
    return dt * kl_est # importance weight coming from uniform sampling


def reconstruction_loss(key: jr.PRNGKey, ctx_seq: jnp.array, process_ctx: Callable[[jnp.array, jnp.array], jnp.array], params: Dict[str, jnp.array], post_net: Posterior, ys: jnp.array, obs_times: jnp.array, likelihood: Likelihood, mask: Optional[jnp.array] = None) -> float:
    """
    Negative expected log likelihood under approximate posterior
    """
    key_ell, key_t = jr.split(key, 2)

    # Randomly select an observation among valid observations
    if mask is None:
        valid_len = ys.shape[0]
        idx = jr.randint(key_t, shape=(), minval=0, maxval=valid_len)
    else:
        mask_bool = mask.astype(bool)
        valid_len = jnp.maximum(jnp.sum(mask_bool).astype(jnp.int32), 1)
        valid_idx = jnp.nonzero(mask_bool, size=mask.shape[0], fill_value=0)[0]
        j = jr.randint(key_t, shape=(), minval=0, maxval=valid_len)
        idx = valid_idx[j]

    yt, t = ys[idx], obs_times[idx]
    mt, Rt = post_net.apply(params["posterior_params"], jnp.array([t]), ctx_seq, process_ctx)
    return (-1.0) * valid_len * likelihood.ell(yt, t, mt, Rt @ Rt.T, key_ell, params["output_params"]) # importance weight coming from uniform sampling


def prior_loss(ctx_seq: jnp.array, process_ctx: Callable[[jnp.array, jnp.array], jnp.array], params: Dict[str, jnp.array], post_net: Posterior, jitter: float = 1e-8) -> float:
    """
    KL between initial distribution of approximate posterior and prior KL(q(x(0))||p(x(0)))
    NOTE: the prior loss is always computed at time 0, irrespective of the mask
    """
    m0, R0 = post_net.apply(params["posterior_params"], jnp.zeros((1), dtype=ctx_seq.dtype), ctx_seq, process_ctx)
    init_params = params["init_params"]
    return gaussian_kl(m0, R0 @ R0.T, init_params["mu0"], init_params["V0"], jitter=jitter)


def make_step(ys: jnp.array, obs_times: jnp.array, likelihood: Likelihood, encoder: GRUEncoder, process_ctx: Callable[[jnp.array, jnp.array, jnp.array], jnp.array], post_net: Posterior, prior: SDE, optimizer: optax.GradientTransformation, mc_samples: int = 1, jitter: float = 1e-8, gauge: str = "sqrt", div_free: Optional[HelmCorrection] = None, learn_prior: bool = True, learn_output: bool = True, process_data: Optional[DataProcessor] = None, per_trial_posterior: bool = False, param_map: Optional[Callable[[Dict], Dict]] = None) -> Callable:
    """
    Makes the gradient step
    """

    if process_data is None:
        process_data = process_data_identity

    def step(key, params, opt_state, kl_weight=jnp.array(1.0)):
        key_data, key_kl, key_rec = jr.split(key, 3)
        processed = process_data(ys, obs_times, key_data)
        ys_step, obs_times_step, mask_step = processed.ys, processed.obs_times, processed.mask
        B = ys_step.shape[0]

        def total_loss(trainable):
            params_full = {
                **params,
                "encoder_params": trainable["encoder_params"],
                "posterior_params": trainable["posterior_params"],
            }
            if learn_prior:
                params_full["sde_params"] = trainable["sde_params"]
                params_full["init_params"] = trainable["init_params"]
            if learn_output:
                params_full["output_params"] = trainable["output_params"]
            if param_map is not None:
                params_full = param_map(params_full)

            # Forward pass through encoder on processed data
            # NOTE: this is done on the processed data
            ctx_seq = vmap(lambda y_i, m_i: encoder.apply(params_full["encoder_params"], y_i, m_i))(ys_step, mask_step) # (B, T+1, H)

            # Separate encoder pass for the KL0 term
            # NOTE: KL0 is always evaluated at time 0 in the unprocessed dataset
            if processed.mask_kl0 is not None:
                mask_kl0 = processed.mask_kl0
            else:
                mask_kl0 = jnp.ones(ys.shape[:2], dtype=ys.dtype)
            ctx_seq_full = vmap(lambda y_i, m_i: encoder.apply(params_full["encoder_params"], y_i, m_i))(ys, mask_kl0)

            # Compute path-space KL loss
            if processed.kl_t0 is not None and processed.kl_t1 is not None:
                kl_t0_batch = processed.kl_t0
                kl_t1_batch = processed.kl_t1
            else:
                # Default: integrate over the time interval determined by the first and last unmasked positions
                mask_bool = mask_step.astype(bool)
                kl_t0_batch = vmap(lambda obs_t, mb: obs_t[jnp.argmax(mb).astype(jnp.int32)])(obs_times_step, mask_bool) # first unmasked position
                kl_t1_batch = vmap(lambda obs_t, mb: obs_t[mb.shape[0] - 1 - jnp.argmax(mb[::-1]).astype(jnp.int32)])(obs_times_step, mask_bool) # last unmasked position

            keys_kl = jr.split(key_kl, B * mc_samples).reshape(B, mc_samples, -1)
            if per_trial_posterior:
                def _kl_loss(key_i, ctx_seq_i, obs_times_i, mask_i, t0_i, t1_i, post_params_i):
                    proc_ctx = partial(process_ctx, obs_times_i, mask=mask_i)
                    params_i = {**params_full, "posterior_params": post_params_i}
                    return kl_loss(key_i, ctx_seq_i, proc_ctx, params_i, post_net, t0_i, t1_i, prior, gauge=gauge, div_free=div_free, jitter=jitter)
                kl = (vmap(vmap(_kl_loss, in_axes=(0, None, None, None, None, None, None)),in_axes=(0, 0, 0, 0, 0, 0, 0))(keys_kl, ctx_seq, obs_times_step, mask_step, kl_t0_batch, kl_t1_batch, params_full["posterior_params"])).mean()
            else:
                def _kl_loss(key_i, ctx_seq_i, obs_times_i, mask_i, t0_i, t1_i):
                    proc_ctx = partial(process_ctx, obs_times_i, mask=mask_i)
                    return kl_loss(key_i, ctx_seq_i, proc_ctx, params_full, post_net, t0_i, t1_i, prior, gauge=gauge, div_free=div_free, jitter=jitter)
                kl = (vmap(vmap(_kl_loss, in_axes=(0, None, None, None, None, None)), in_axes=(0, 0, 0, 0, 0, 0))(keys_kl, ctx_seq, obs_times_step, mask_step, kl_t0_batch, kl_t1_batch)).mean() # ()

            # Compute reconstruction loss
            keys_rec = jr.split(key_rec, B * mc_samples).reshape(B, mc_samples, -1)
            if per_trial_posterior:
                def _reconstruction_loss(key_i, ctx_seq_i, obs_times_i, ys_i, mask_i, post_params_i):
                    proc_ctx = partial(process_ctx, obs_times_i, mask=mask_i)
                    params_i = {**params_full, "posterior_params": post_params_i}
                    return reconstruction_loss(key_i, ctx_seq_i, proc_ctx, params_i, post_net, ys_i, obs_times_i, likelihood, mask=mask_i)
                rec = (vmap(vmap(_reconstruction_loss, in_axes=(0, None, None, None, None, None)), in_axes=(0, 0, 0, 0, 0, 0))(keys_rec, ctx_seq, obs_times_step, ys_step, mask_step, params_full["posterior_params"])).mean()
            else:
                def _reconstruction_loss(key_i, ctx_seq_i, obs_times_i, ys_i, mask_i):
                    proc_ctx = partial(process_ctx, obs_times_i, mask=mask_i)
                    return reconstruction_loss(key_i, ctx_seq_i, proc_ctx, params_full, post_net, ys_i, obs_times_i, likelihood, mask=mask_i)
                rec = (vmap(vmap(_reconstruction_loss, in_axes=(0, None, None, None, None)), in_axes=(0, 0, 0, 0, 0))(keys_rec, ctx_seq, obs_times_step, ys_step, mask_step)).mean() # ()

            # Compute initial KL loss
            B_kl0 = ys.shape[0]
            init_params_batch = jax.tree.map(lambda x: jnp.broadcast_to(x[None], (B_kl0,) + x.shape), params_full["init_params"])
            if per_trial_posterior:
                def _prior_loss(ctx_seq_i, obs_times_i, mask_i, init_params_i, post_params_i):
                    params_i = {**params_full, "init_params": init_params_i, "posterior_params": post_params_i}
                    proc_ctx = partial(process_ctx, obs_times_i, mask=mask_i)
                    return prior_loss(ctx_seq_i, proc_ctx, params_i, post_net, jitter=jitter)
                kl0 = vmap(_prior_loss)(ctx_seq_full, obs_times, mask_kl0, init_params_batch, params_full["posterior_params"]).mean()
            else:
                def _prior_loss(ctx_seq_i, obs_times_i, mask_i, init_params_i):
                    params_i = {**params_full, "init_params": init_params_i}
                    proc_ctx = partial(process_ctx, obs_times_i, mask=mask_i)
                    return prior_loss(ctx_seq_i, proc_ctx, params_i, post_net, jitter=jitter)
                kl0 = vmap(_prior_loss)(ctx_seq_full, obs_times, mask_kl0, init_params_batch).mean()

            loss = kl_weight * (kl + kl0) + rec
            return loss, (kl, rec, kl0)

        # Compute gradient step
        trainable = {
            "encoder_params": params["encoder_params"],
            "posterior_params": params["posterior_params"],
        } # parameters to optimize via SGD
        if learn_prior:
            trainable["sde_params"] = params["sde_params"]
            trainable["init_params"] = params["init_params"]
        if learn_output:
            trainable["output_params"] = params["output_params"]
        (loss, (kl, rec, kl0)), grads = jax.value_and_grad(total_loss, has_aux=True)(trainable)
        if per_trial_posterior: # if have per-trial posterior, ensures gradients are O(1)
            grads = {k: (jax.tree.map(lambda g: g * B, v) if k == "posterior_params" else v) for k, v in grads.items()}
        updates, opt_state = optimizer.update(grads, opt_state, trainable)
        trainable = optax.apply_updates(trainable, updates)

        # Update trainable parameters
        params = {
            **params,
            "encoder_params": trainable["encoder_params"],
            "posterior_params": trainable["posterior_params"],
        }
        if learn_prior:
            params["sde_params"] = trainable["sde_params"]
            params["init_params"] = trainable["init_params"]
        if learn_output:
            params["output_params"] = trainable["output_params"]

        # Record metrics
        metrics = {
            "loss": loss,
            "kl": kl,
            "rec": rec,
            "prior": kl0,
        }
        return params, opt_state, metrics
    return step


def train(
    key: jr.PRNGKey, # random key
    ys: jnp.array, # observations (B, T, D)
    obs_times: jnp.array, # times at which observations occur (B, T)
    likelihood: Likelihood, # likelihood model
    output_params: Dict[str, jnp.array], # parameters of the likelihood model
    t_max: float, # max time
    encoder: Union[NullEncoder, GRUEncoder], # encodes observations
    process_ctx: Callable, # processes observational encodings
    post_net: Posterior, # posterior SDE
    prior: SDE, # prior SDE
    sde_params: Dict[str, Any], # prior SDE parameters
    init_params: Dict[str, jnp.array], # initial mean, covariance of prior
    learning_rate_init: float = 1e-3, # initial learning rate
    learning_rate_final: float = 1e-3, # final learning rate; NOTE: equal to learning_rate_init by default, so the default schedule is constant
    mc_samples: int = 1, # number of Monte Carlo samples used for approximating KL, reconstruction losses
    jitter: float = 1e-8, # jitter
    n_iters: int = 5000, # number of iterations
    gauge: str = "sqrt", # choice of reference posterior drift; choices ["sqrt", "sym"]
    div_free: Optional[HelmCorrection] = DEFAULT_CORRECTION, # divergence-free correction to the posterior drift; None runs SDE Matching with no correction
    learn_prior: bool = True, # boolean, whether to learn the prior
    learn_output: bool = True, # boolean, whether to learn the output parameters
    disable_pbar: bool = False, # boolean, whether to disable the progress bar
    grad_clip_norm: Optional[float] = 5.0, # global clipping on gradient norm
    process_data: Optional[DataProcessor] = None, # preprocessing function for observations
    kl_weight: float = 1.0, # weight on KL term (= 1 yields the ELBO)
    kl_anneal_iters: int = 0, # annealing schedule on the KL term
    per_trial_posterior: bool = False, # if True, each trial gets its own posterior params
    param_map: Optional[Callable[[Dict], Dict]] = None, # transforms params_full inside total_loss (for parameter sharing)
) -> Tuple[Dict[str, Any], Dict[str, jnp.array]]:
    """
    Training loop for performing simulation-free inference and learning in the latent SDE model
    """

    if gauge not in ["sqrt", "sym"]:
        raise ValueError("Only gauges [sqrt, sym] are supported")
    if div_free is not None and not isinstance(div_free, HelmCorrection):
        raise TypeError(f"div_free must be a HelmCorrection or None, got {type(div_free).__name__}")
    if process_data is None:
        process_data = process_data_identity

    # Instantiate neural network objects
    y_init, obs_times_init = ys[0], obs_times[0]
    mask_init = jnp.ones(y_init.shape[0], dtype=y_init.dtype)
    key, key_enc, key_post = jr.split(key, 3)

    ## Posterior encoder
    encoder_params = encoder.init(key_enc, y_init, mask_init)

    ## Posterior
    ctx0 = encoder.apply(encoder_params, y_init, mask_init)
    if per_trial_posterior: # initialize per-trial posterior params
        B = ys.shape[0]
        keys_post = jr.split(key_post, B)
        posterior_params = vmap(lambda k: post_net.init(k, jnp.array([t_max / 2]), ctx0, partial(process_ctx, obs_times_init, mask=mask_init)))(keys_post)
    else:
        posterior_params = post_net.init(key_post, jnp.array([t_max / 2]), ctx0, partial(process_ctx, obs_times_init, mask=mask_init))

    params = {
        "encoder_params": encoder_params,
        "posterior_params": posterior_params,
        "sde_params": sde_params,
        "output_params": output_params,
        "init_params": init_params,
    }
    trainable = {
        "encoder_params": params["encoder_params"],
        "posterior_params": params["posterior_params"],
    }
    if learn_prior: # learn the parameters of the prior (initial distribution, drift parameters)
        trainable["sde_params"] = params["sde_params"]
        trainable["init_params"] = params["init_params"]
    if learn_output: # learn the parameters of the output model
        trainable["output_params"] = params["output_params"]

    # Use exponential lr schedule
    weight_decay = 0.0
    if n_iters <= 1:
        decay_rate = 1.0 # constant lr; a single step has nothing to decay over
    else:
        decay_rate = (learning_rate_final / learning_rate_init) ** (1.0 / (n_iters - 1))
    lr_schedule = optax.exponential_decay(
        init_value=learning_rate_init,
        transition_steps=1,
        decay_rate=decay_rate,
        staircase=False,
    ) # exponential lr decay

    if grad_clip_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(grad_clip_norm),
            optax.adamw(
                learning_rate=lr_schedule,
                weight_decay=weight_decay,
            ),
        )
    else:
        optimizer = optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=weight_decay,
        ) # NOTE: adamw with weight_decay = 0 is the same as adam
    opt_state = optimizer.init(trainable)

    step = make_step(
        ys,
        obs_times,
        likelihood,
        encoder,
        process_ctx,
        post_net,
        prior,
        optimizer,
        mc_samples = mc_samples,
        jitter = jitter,
        gauge = gauge,
        div_free = div_free,
        learn_prior = learn_prior,
        learn_output = learn_output,
        process_data = process_data,
        per_trial_posterior = per_trial_posterior,
        param_map = param_map,
    )
    step = jax.jit(step) # jit the step function

    metrics_log = []
    # Run the optimization
    pbar = tqdm.tqdm(range(n_iters), disable=disable_pbar)
    for i in pbar:
        subkey = jr.fold_in(key, i)
        if kl_anneal_iters > 0:
            kl_w = kl_weight * min(1.0, i / max(1, kl_anneal_iters))
        else:
            kl_w = kl_weight
        params, opt_state, metrics = step(subkey, params, opt_state, kl_weight=jnp.array(kl_w))
        metrics_log.append(metrics)

        if not disable_pbar:
            pbar.set_postfix(loss=float(metrics["loss"]), kl=float(metrics["kl"]), rec=float(metrics["rec"]), prior=float(metrics["prior"]), kl_w=round(kl_w, 4))

    metrics_arr = {k: jnp.stack([m[k] for m in metrics_log]) for k in metrics_log[0].keys()}
    return params, metrics_arr
