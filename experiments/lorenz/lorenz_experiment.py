"""
Reproduces the noisy Lorenz attractor experiment from the paper
"""

import argparse
import csv
import logging
import os
import pickle
import sys
from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import vmap

jax.config.update("jax_enable_x64", True)

from helmholtz_sde.sde import NeuralSDE, DriftNetwork, DiffusionNetwork
from helmholtz_sde.posterior.encoder import weighted_ctx, ForwardGRUEncoder
from helmholtz_sde.posterior.nn_posterior import InferenceNetwork
from helmholtz_sde.posterior.drift import apply_inference_net_time_derivs
from helmholtz_sde.train import train
from helmholtz_sde.utils.general_helpers import simulate_sde, simulate_learned_prior_samples
from helmholtz_sde.utils.plotting import time_to_index

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) # shared experiment code
from experiment_utils import build_correction, eval_correction, batched_eval_elbo
from lorenz_utils import (
    GaussianFixedVariance,
    LorenzAttractor,
    compute_lorenz_fixed_points,
    compute_marginal_kde_kl,
    global_correlation_error,
    posterior_true_logprob,
    summary_stats,
    within_lobe_correlation_error,
)

logger = logging.getLogger(__name__)


# --------------------- Evaluation functions ---------------------
def evaluate_prior_forecast_error(key: jr.PRNGKey, ys_obs: jnp.array, obs_times: jnp.array, ys_future: jnp.array, del_future: float, n_future_timesteps: int, prior: NeuralSDE, encoder: ForwardGRUEncoder, post_net: InferenceNetwork, params: Dict[str, Any], process_ctx: Callable, n_forecast_samples: int = 256) -> Tuple[jnp.array, jnp.array]:
    """
    Evaluate forecasting error obtained by sampling the approximate posterior at the terminal time
    and simulating the learned prior SDE
    """
    B = ys_obs.shape[0]
    t_anchor = obs_times[0, -1]
    C, d = params["output_params"]["C"], params["output_params"]["d"]

    # Terminal posterior marginal q(x(T) | y_{1:n}) of each trial
    def _final_marginal(ys: jnp.array, obs_times_b: jnp.array) -> Tuple[jnp.array, jnp.array]:
        ctx_seq = encoder.apply(params["encoder_params"], ys)
        proc_ctx = partial(process_ctx, obs_times_b)
        t_arr = jnp.array([t_anchor], dtype=ys.dtype)
        mt, Rt, _, _ = apply_inference_net_time_derivs(post_net, params["posterior_params"], t_arr, ctx_seq, proc_ctx)
        return mt, Rt

    ms_T, Rs_T = vmap(_final_marginal)(ys_obs, obs_times) # (B, K), (B, K, K)
    K = ms_T.shape[-1]

    # Sample x(T) from each terminal marginal and roll the learned prior forward
    def _sample_xT(key_b: jr.PRNGKey, m: jnp.array, R: jnp.array) -> jnp.array:
        eps = jr.normal(key_b, shape=(n_forecast_samples, K), dtype=m.dtype)
        return m[None, :] + eps @ R.T

    def _simulate_trial(key_b: jr.PRNGKey, x0s: jnp.array) -> jnp.array:
        sim_keys = jr.split(key_b, n_forecast_samples)
        return vmap(partial(simulate_sde, sde=prior, sde_params=params["sde_params"], t_max=del_future, n_timesteps=n_future_timesteps))(sim_keys, x0s)

    key_init, key_roll = jr.split(key)
    xT_samples = vmap(_sample_xT)(jr.split(key_init, B), ms_T, Rs_T) # (B, n_samples, K)
    xs_fore = vmap(_simulate_trial)(jr.split(key_roll, B), xT_samples) # (B, n_samples, T_future + 1, K)
    ys_fore = vmap(vmap(vmap(lambda x: C @ x + d)))(xs_fore) # (B, n_samples, T_future + 1, D)
    ys_fore_mean = jnp.mean(ys_fore, axis=1) # (B, T_future + 1, D), predictive mean

    # Squared errors at the future times (anchor time is excluded)
    mean_sqerr = jnp.sum(jnp.square(ys_fore_mean - ys_future), axis=-1)[:, 1:] # (B, T_future)
    pred_sqerr = jnp.mean(jnp.sum(jnp.square(ys_fore - ys_future[:, None]), axis=-1), axis=1)[:, 1:] # (B, T_future)
    return jnp.mean(mean_sqerr), jnp.mean(pred_sqerr)


# --------------------- Main function ---------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for name in ("jax", "jax._src.xla_bridge", "jaxlib"): # silence JAX backend warnings
        logging.getLogger(name).setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Noisy Lorenz attractor experiment")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)

    # Dataset generation (ignored when --dataset_path is passed)
    parser.add_argument("--t_max", type=float, default=5.0)
    parser.add_argument("--noise_scale", type=float, default=5.0)

    # Observations
    parser.add_argument("--obs_noise_scale", type=float, default=0.3)
    parser.add_argument("--obs_per_t", type=int, default=4)
    parser.add_argument("--n_trials", type=int, default=1024)
    parser.add_argument("--n_trials_test", type=int, default=2048)

    # Model and training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--embedding_size", type=int, default=100)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--n_iters", type=int, default=30000)
    parser.add_argument("--n_reps", type=int, default=5)
    parser.add_argument("--learn_diff", type=int, choices=[0, 1], default=1)
    parser.add_argument("--gauge", type=str, choices=["sqrt", "sym"], default="sqrt")
    parser.add_argument("--div_free", type=str, default="none", choices=["none", "taylor", "least_squares"])
    parser.add_argument("--ell", type=int, default=1)
    parser.add_argument("--kappa", type=int, default=1)
    parser.add_argument("--n_mc", type=int, default=1)
    parser.add_argument("--n_nodes", type=int, default=None)

    # Evaluation
    parser.add_argument("--n_nodes_eval", type=int, default=6)
    parser.add_argument("--n_eval_prior_samples", type=int, default=2048)
    parser.add_argument("--corr_max_time", type=float, default=2.0)
    parser.add_argument("--corr_skip", type=int, default=50)
    parser.add_argument("--lobe_min_count", type=int, default=50)
    parser.add_argument("--fit_radius_quantile", type=float, default=0.95)
    parser.add_argument("--kde_time_stride", type=int, default=100)
    parser.add_argument("--kde_bandwidth_splits", type=int, default=3)
    parser.add_argument("--kde_outer_splits", type=int, default=10)
    parser.add_argument("--posterior_eval_skip", type=int, default=10)
    parser.add_argument("--posterior_eval_batch_size", type=int, default=64)
    parser.add_argument("--del_future", type=float, default=1.)
    parser.add_argument("--n_forecast_samples", type=int, default=64)
    args = parser.parse_args()

    # Parse arguments
    (
        output_dir,
        output_name,
        dataset_path,
        t_max,
        sigma,
        ssigma,
        obs_per_t,
        n_trials,
        n_trials_test,
        lr,
        hidden_dim,
        embedding_size,
        depth,
        n_iters,
        n_reps,
        learn_diff,
        gauge,
        div_free,
        ell,
        kappa,
        n_mc,
        n_nodes,
        n_nodes_eval,
        n_eval_prior_samples,
        corr_max_time,
        corr_skip,
        lobe_min_count,
        fit_radius_quantile,
        kde_time_stride,
        kde_bandwidth_splits,
        kde_outer_splits,
        posterior_eval_skip,
        posterior_eval_batch_size,
        del_future,
        n_forecast_samples,
    ) = (
        args.output_dir,
        args.output_name,
        args.dataset_path,
        args.t_max,
        args.noise_scale,
        args.obs_noise_scale,
        args.obs_per_t,
        args.n_trials,
        args.n_trials_test,
        args.lr,
        args.hidden_dim,
        args.embedding_size,
        args.depth,
        args.n_iters,
        args.n_reps,
        args.learn_diff,
        args.gauge,
        args.div_free,
        args.ell,
        args.kappa,
        args.n_mc,
        args.n_nodes,
        args.n_nodes_eval,
        args.n_eval_prior_samples,
        args.corr_max_time,
        args.corr_skip,
        args.lobe_min_count,
        args.fit_radius_quantile,
        args.kde_time_stride,
        args.kde_bandwidth_splits,
        args.kde_outer_splits,
        args.posterior_eval_skip,
        args.posterior_eval_batch_size,
        args.del_future,
        args.n_forecast_samples,
    )
    learn_diff = bool(learn_diff)
    div_free = build_correction(div_free, ell, kappa, n_mc, n_nodes)
    if learn_diff and div_free is not None:
        raise ValueError("--learn_diff 1 learns a state-dependent diffusion coefficient, which the Helmholtz correction does not support; pass --learn_diff 0 or --div_free none")
    div_free_eval = eval_correction(div_free, n_nodes=n_nodes_eval)
    logger.info(f"Gauge = {gauge}, Helmholtz correction = {div_free} (ELBO evaluated with {div_free_eval}), learn_diff = {learn_diff}")

    # Dataset
    K, D = 4, 3 # latent and observation dimension
    sde_true = LorenzAttractor(D)
    if dataset_path is not None: # load an existing Lorenz dataset
        with open(os.path.join(dataset_path, "train.pkl"), "rb") as f:
            blob_train = pickle.load(f)
        with open(os.path.join(dataset_path, "test.pkl"), "rb") as f:
            xs_test = jnp.array(pickle.load(f))
        xs_train = jnp.array(blob_train["xs"])
        t_grid = jnp.array(blob_train["t_grid"])
        t_max, n_timesteps = t_grid[-1].item(), t_grid.shape[0] - 1 # NOTE: --t_max and --noise_scale are overridden by the values the dataset was generated with
        a = jnp.array(blob_train["sde_params"])
        sigma = blob_train["sigma"]
        means, stds = jnp.array(blob_train["means"]), jnp.array(blob_train["stds"])
        sde_params_true = {"a": a, "G": sigma * jnp.eye(D)}
        logger.info(f"Loaded Lorenz dataset from {dataset_path}")
    else: # generate the Lorenz dataset
        a = jnp.array([10., 28., 8. / 3.])
        sde_params_true = {"a": a, "G": sigma * jnp.eye(D)}
        n_timesteps = int(t_max * 1000) # use grid of size dt = 0.001
        t_grid = jnp.linspace(0., t_max, n_timesteps + 1)
        n_train, n_test = 1024, 2048
        init_key, sample_key = jr.split(jr.PRNGKey(0), 2)
        x0 = jr.normal(init_key, shape=(n_train + n_test, D)) # N(0, I) initialization in x space
        xs = vmap(partial(simulate_sde, sde=sde_true, sde_params=sde_params_true, t_max=t_max, n_timesteps=n_timesteps))(jr.split(sample_key, n_train + n_test), x0)
        means = jnp.mean(jnp.reshape(xs, (-1, D)), axis=0)
        stds = jnp.std(jnp.reshape(xs, (-1, D)), axis=0)
        xs_train, xs_test = xs[:n_train], xs[n_train:]

        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "lorenz") # the directory lorenz_attractor.ipynb writes to
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "train.pkl"), "wb") as f:
            pickle.dump({"xs": np.asarray(xs_train), "t_grid": np.asarray(t_grid), "sde_params": np.asarray(a), "sigma": sigma, "means": np.asarray(means), "stds": np.asarray(stds)}, f)
        with open(os.path.join(save_dir, "test.pkl"), "wb") as f:
            pickle.dump(np.asarray(xs_test), f)
        logger.info(f"Saved Lorenz dataset to {save_dir}")
    n_train, n_test = xs_train.shape[0], xs_test.shape[0]

    if n_trials > n_train:
        raise ValueError(f"Requested n_trials={n_trials}, but dataset only has n_train={n_train}")
    if n_trials_test > n_test:
        raise ValueError(f"Requested n_trials_test={n_trials_test}, but dataset only has n_test={n_test}")
    xs_train, xs_test = xs_train[:n_trials], xs_test[:n_trials_test] # use the first trials of each split
    dt = float(t_grid[1] - t_grid[0])

    # Observations are the standardized latents, observed on a regular grid of n_obs times with Gaussian noise
    C_true, d_true = jnp.diag(1 / stds), -means / stds
    apply_affine_true = vmap(vmap(lambda x: C_true @ x + d_true))
    ys_train, ys_test = apply_affine_true(xs_train), apply_affine_true(xs_test)

    n_obs = int(t_max * obs_per_t) + 1 # add 1 for the endpoint
    if n_obs == 1:
        idx_obs = jnp.array([0])
    else:
        idx_obs = jnp.array([time_to_index(t, t_max, n_timesteps + 1) for t in (t_max / (n_obs - 1)) * jnp.arange(n_obs)])
    obs_times_train = jnp.broadcast_to(t_grid[idx_obs], (n_trials, n_obs))
    obs_times_test = jnp.broadcast_to(t_grid[idx_obs], (n_trials_test, n_obs))
    obs_key_train, obs_key_test = jr.split(jr.PRNGKey(1), 2)
    ys_obs_train = ys_train[:, idx_obs] + ssigma * jr.normal(obs_key_train, shape=(n_trials, n_obs, D))
    ys_obs_test = ys_test[:, idx_obs] + ssigma * jr.normal(obs_key_test, shape=(n_trials_test, n_obs, D))

    # Model
    likelihood = GaussianFixedVariance(ssigma=ssigma)
    process_ctx = partial(weighted_ctx, beta=n_obs)
    encoder = ForwardGRUEncoder(embedding_size)
    post_net = InferenceNetwork(hidden_dim=hidden_dim, K=K, depth=depth, jitter=1e-8)
    drift_net = DriftNetwork(hidden_dim=hidden_dim, K=K, depth=1, include_time=False)
    if learn_diff: # learned diagonal, state-dependent diffusion coefficient (Bartosh et al., 2025)
        diff_net = DiffusionNetwork(hidden_dim=hidden_dim, K=K, depth=1, include_time=False)
        prior = NeuralSDE(K, apply_fn_drift=drift_net.apply, apply_fn_diffusion=diff_net.apply, div_GGt_apply_fn=lambda params, x, t: diff_net.apply(params, x, t, method=DiffusionNetwork.div_GGt))
    else: # diffusion coefficient fixed to the identity
        prior = NeuralSDE(K, apply_fn_drift=drift_net.apply, apply_fn_diffusion=None)
    init_params = {"mu0": jnp.zeros((K,)), "V0": jnp.eye(K)}

    # Lobe centers of the attractor in observation coordinates, used for computing within-lobe correlations
    centers = compute_lorenz_fixed_points(means, stds, rho=float(a[1]), beta=float(a[2]))

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, output_name + ".csv")
    raw_path = os.path.join(output_dir, output_name + "_raw.pkl")

    rep_metrics: List[Dict[str, float]] = []
    train_keys = jr.split(jr.PRNGKey(2), n_reps)
    for rep, key_i in enumerate(train_keys):
        logger.info(f"Starting replicate {rep + 1}/{n_reps}")

        # Initialize the prior and the output map
        key_i, init_key = jr.split(key_i, 2)
        init_key_drift, init_key_diff, init_key_output = jr.split(init_key, 3)
        network_params_drift = drift_net.init(init_key_drift, jnp.zeros((K,)), jnp.zeros((1,)))
        network_params_diff = diff_net.init(init_key_diff, jnp.zeros((K,)), jnp.zeros((1,))) if learn_diff else None
        sde_params = {"network_params_drift": network_params_drift, "network_params_diffusion": network_params_diff}
        C_init_key, d_init_key = jr.split(init_key_output, 2)
        output_params_init = {"C": jr.normal(C_init_key, shape=(D, K)), "d": jr.normal(d_init_key, shape=(D,))}

        # Train
        key_i, key_train = jr.split(key_i, 2)
        params, _ = train(
            key_train,
            ys_obs_train,
            obs_times_train,
            likelihood,
            output_params_init,
            t_max,
            encoder,
            process_ctx,
            post_net,
            prior,
            sde_params,
            init_params,
            learning_rate_init=lr,
            learning_rate_final=lr,
            mc_samples=1,
            jitter=1e-8,
            n_iters=n_iters,
            div_free=div_free,
            learn_prior=True,
            learn_output=True,
            gauge=gauge,
            grad_clip_norm=5.0,
            disable_pbar=True,
        )

        # Samples from the learned prior, in observation space
        key_i, key_eval = jr.split(key_i, 2)
        xs_eval = simulate_learned_prior_samples(key_eval, prior, params, t_max=t_max, n_timesteps=n_timesteps, n_eval=n_eval_prior_samples)
        C_hat, d_hat = params["output_params"]["C"], params["output_params"]["d"]
        ys_eval = vmap(vmap(lambda x: C_hat @ x + d_hat))(xs_eval)
        ys_test_np, ys_eval_np = np.asarray(ys_test), np.asarray(ys_eval)

        # Time-lagged correlation errors, globally and within each lobe
        global_corr_err = global_correlation_error(ys_test_np, ys_eval_np, dt=dt, max_time=corr_max_time, skip=corr_skip)
        lobe_corr_err = within_lobe_correlation_error(ys_test_np, ys_eval_np, centers=centers, dt=dt, max_time=corr_max_time, skip=corr_skip, min_count=lobe_min_count, fit_radius_quantile=fit_radius_quantile)

        # Marginal KL divergences between the true and the learned prior
        times = np.arange(0, ys_test.shape[1], kde_time_stride)
        marginal_kl_pq = compute_marginal_kde_kl(ys_test_np, ys_eval_np, times, n_bandwidth_splits=kde_bandwidth_splits, n_outer_splits=kde_outer_splits)
        marginal_kl_qp = compute_marginal_kde_kl(ys_eval_np, ys_test_np, times, n_bandwidth_splits=kde_bandwidth_splits, n_outer_splits=kde_outer_splits)

        # Held-out log probability of the true latent paths under the posterior
        skip = posterior_eval_skip
        posterior_logprob = posterior_true_logprob(ys_obs=ys_obs_train, obs_times=obs_times_train, ys_true=ys_train[:, ::skip], t_grid=t_grid[::skip], encoder=encoder, post_net=post_net, params=params, process_ctx=process_ctx, ssigma=ssigma, batch_size=posterior_eval_batch_size)

        # Forecasting error: simulate true trajectories past t_max, then forecast them from the terminal posterior marginals
        n_future_timesteps = int(round(del_future / dt))
        simulate_future = lambda key, x0s: vmap(partial(simulate_sde, sde=sde_true, sde_params=sde_params_true, t_max=del_future, n_timesteps=n_future_timesteps))(jr.split(key, x0s.shape[0]), x0s)
        key_i, sample_key_train, sample_key_test = jr.split(key_i, 3)
        ys_future_train = apply_affine_true(simulate_future(sample_key_train, xs_train[:, -1]))
        ys_future_test = apply_affine_true(simulate_future(sample_key_test, xs_test[:, -1]))

        key_i, forecast_key_train, forecast_key_test = jr.split(key_i, 3)
        forecast_kwargs = dict(del_future=del_future, n_future_timesteps=n_future_timesteps, prior=prior, encoder=encoder, post_net=post_net, params=params, process_ctx=process_ctx, n_forecast_samples=n_forecast_samples)
        forecast_mean_mse_train, forecast_pred_mse_train = evaluate_prior_forecast_error(forecast_key_train, ys_obs_train, obs_times_train, ys_future_train, **forecast_kwargs)
        forecast_mean_mse_test, forecast_pred_mse_test = evaluate_prior_forecast_error(forecast_key_test, ys_obs_test, obs_times_test, ys_future_test, **forecast_kwargs)

        # ELBO on the training observations, with the divergence-free correction computed using quadrature
        nelbo, kl_term, rec_term, prior_term = batched_eval_elbo(
            eval_key=key_i,
            ys_obs=ys_obs_train,
            obs_times=obs_times_train,
            encoder=encoder,
            post_net=post_net,
            params=params,
            prior=prior,
            likelihood=likelihood,
            process_ctx=process_ctx,
            t_max=t_max,
            gauge=gauge,
            div_free=div_free_eval,
            batch_size=posterior_eval_batch_size,
        )

        rep_metric = {
            "rep": rep,
            "global_corr_err": global_corr_err,
            "lobe_corr_err": lobe_corr_err,
            "marginal_kl_pq": marginal_kl_pq,
            "marginal_kl_qp": marginal_kl_qp,
            "posterior_true_logprob": posterior_logprob,
            "nelbo": float(nelbo),
            "elbo_kl_term": float(kl_term),
            "elbo_reconstruction_term": float(rec_term),
            "elbo_prior_term": float(prior_term),
            "forecast_pred_mse_train": float(forecast_pred_mse_train),
            "forecast_pred_mse_test": float(forecast_pred_mse_test),
            "forecast_mean_mse_train": float(forecast_mean_mse_train),
            "forecast_mean_mse_test": float(forecast_mean_mse_test),
        }
        logger.info(f"Replicate {rep + 1}/{n_reps}: {rep_metric}")
        rep_metrics.append(rep_metric)

    # Summary statistics across replicates
    summary = {}
    for name in rep_metrics[0]:
        if name != "rep":
            summary.update(summary_stats([m[name] for m in rep_metrics], name))

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(summary.keys()))
        writer.writerow(list(summary.values()))
    with open(raw_path, "wb") as f:
        pickle.dump({"summary": summary, "rep_metrics": rep_metrics}, f)
    logger.info(f"Wrote summary CSV to {csv_path}")
    logger.info(f"Wrote raw replicate metrics to {raw_path}")


if __name__ == "__main__":
    main()
