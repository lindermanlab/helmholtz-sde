"""
Reproduces the Ornstein-Uhlenbeck spiral inference experiment from the paper (fixed prior and output model)
"""

import argparse
import csv
import itertools
import logging
import os
import pickle
import sys
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

jax.config.update("jax_enable_x64", True)

from helmholtz_sde.sde import LinearSDE
from helmholtz_sde.likelihood import Gaussian
from helmholtz_sde.posterior.encoder import NullEncoder, constant_ctx
from helmholtz_sde.posterior.nn_posterior import InferenceNetwork
from helmholtz_sde.posterior.grid_posterior import GridInferenceNetwork
from helmholtz_sde.posterior.kernel_glm_posterior import GPInferenceNetwork, init_gp_posterior, share_kernel_hyperparameters
from helmholtz_sde.train import train
from helmholtz_sde.data import process_data_full_kl
from helmholtz_sde.utils.general_helpers import inverse_softplus, sym_sqrt_and_invsqrt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) # shared experiment code
from experiment_utils import batched_eval_elbo, build_correction, eval_correction
from spiral_learning import eval_posterior

logger = logging.getLogger(__name__)


# --------------------- Main function ---------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for name in ("jax", "jax._src.xla_bridge", "jaxlib"): # silence JAX backend warnings
        logging.getLogger(name).setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Ornstein-Uhlenbeck spiral inference experiment")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, required=True)

    # Observations
    parser.add_argument("--n_trials", type=int, default=16)
    parser.add_argument("--ssigmas", type=float, nargs="+", default=[0.05, 0.3, 1.0])
    parser.add_argument("--n_obs_grid", type=int, nargs="+", default=[5, 10, 20, 50])

    # Method, posterior and training
    parser.add_argument("--gauge", type=str, choices=["sqrt", "sym"], default="sqrt")
    parser.add_argument("--div_free", type=str, default="none", choices=["none", "taylor", "least_squares"])
    parser.add_argument("--ell", type=int, default=1)
    parser.add_argument("--kappa", type=int, default=1)
    parser.add_argument("--n_mc", type=int, default=1)
    parser.add_argument("--n_nodes", type=int, default=None)
    parser.add_argument("--posterior_type", type=str, default="grid", choices=["grid", "gp", "nn"])
    parser.add_argument("--grid_size", type=int, default=100)
    parser.add_argument("--n_tau", type=int, default=200)
    parser.add_argument("--shared_kernel_params", action="store_true", default=False)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_final", type=float, default=1e-4)
    parser.add_argument("--mc_samples", type=int, default=1001)
    parser.add_argument("--n_iters", type=int, default=30000)

    # Evaluation
    parser.add_argument("--n_nodes_eval", type=int, default=4)
    args = parser.parse_args()

    # Parse arguments
    (
        dataset_dir,
        output_dir,
        output_name,
        n_trials,
        ssigmas,
        n_obs_grid,
        gauge,
        div_free,
        ell,
        kappa,
        n_mc,
        n_nodes,
        posterior_type,
        grid_size,
        n_tau,
        shared_kernel_params,
        hidden_dim,
        depth,
        lr,
        lr_final,
        mc_samples,
        n_iters,
        n_nodes_eval,
    ) = (
        args.dataset_dir,
        args.output_dir,
        args.output_name,
        args.n_trials,
        args.ssigmas,
        args.n_obs_grid,
        args.gauge,
        args.div_free,
        args.ell,
        args.kappa,
        args.n_mc,
        args.n_nodes,
        args.posterior_type,
        args.grid_size,
        args.n_tau,
        args.shared_kernel_params,
        args.hidden_dim,
        args.depth,
        args.lr,
        args.lr_final,
        args.mc_samples,
        args.n_iters,
        args.n_nodes_eval,
    )
    div_free = build_correction(div_free, ell, kappa, n_mc, n_nodes)
    if div_free is not None and div_free.ell > 2:
        raise ValueError(f"the KL to the exact posterior supports Helmholtz corrections of degree at most 2, got --ell {ell}")
    div_free_eval = eval_correction(div_free, n_nodes=n_nodes_eval)
    logger.info(f"Gauge = {gauge}, Helmholtz correction = {div_free} (evaluated with {div_free_eval})")

    # Dataset
    with open(os.path.join(dataset_dir, "train.pkl"), "rb") as f:
        blob_train = pickle.load(f)
    xs_train = jnp.array(blob_train["xs"])
    t_grid = jnp.array(blob_train["t_grid"])
    alpha, omega = float(blob_train["sde_params"]["alpha"]), float(blob_train["sde_params"]["kappa"]) # decay coefficient and rotation rate (stored under the key kappa)
    C_true, d_true = jnp.array(blob_train["output_params"]["C"]), jnp.array(blob_train["output_params"]["d"])
    K = xs_train.shape[-1]
    n_timesteps, t_max = t_grid.shape[0] - 1, t_grid[-1].item()
    if n_trials > xs_train.shape[0]:
        raise ValueError(f"Requested n_trials={n_trials}, but the dataset only has {xs_train.shape[0]} training trials")
    xs_train = xs_train[:n_trials]
    logger.info(f"Loaded dataset from {dataset_dir}; using the first {n_trials} training trials")

    # True generative model, fixed throughout
    J = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    sde_true = LinearSDE(K)
    sde_params_true = {"A": -alpha * jnp.eye(K) + omega * J, "b": jnp.zeros((K))}
    init_params_true = {"mu0": jnp.zeros((K)), "V0": (1.0 / (2.0 * alpha)) * jnp.eye(K)} # stationary distribution
    likelihood = Gaussian()
    C_inv = jnp.linalg.inv(C_true)

    # Posterior, one per trial without amortization; the grid and GP posteriors are initialized per setting from the observations
    encoder, process_ctx = NullEncoder(value=0.), constant_ctx
    param_map = None
    if posterior_type == "grid":
        grid_times = jnp.linspace(0.0, t_max, grid_size + 1)
    elif posterior_type == "gp":
        param_map = share_kernel_hyperparameters if shared_kernel_params else None
        logger.info(f"GP posterior with n_tau={n_tau}, shared kernel hyperparameters = {shared_kernel_params}")
    else:
        post_net = InferenceNetwork(hidden_dim=hidden_dim, K=K, depth=depth, jitter=1e-8)

    # Noise-free observations of the training trials, a base noise draw rescaled for each ssigma, and observation times for each n_obs
    ys_clean = vmap(vmap(lambda x: C_true @ x + d_true))(xs_train) # (B, T, D)
    base_noise = jr.normal(jr.PRNGKey(1), shape=(n_trials, n_timesteps + 1, C_true.shape[0])) # (B, T, D)
    idx_root, training_key = jr.split(jr.PRNGKey(2), 2)
    obs_index_cache = {}
    for key_n_obs, n_obs in zip(jr.split(idx_root, len(n_obs_grid)), n_obs_grid):
        _, idx_key = jr.split(key_n_obs, 2)
        obs_index_cache[n_obs] = vmap(lambda key: jnp.sort(jr.choice(key, a=n_timesteps + 1, shape=(n_obs,), replace=False)))(jr.split(idx_key, n_trials)) # (B, n_obs)
    ys_full_cache = {}
    for ssigma in ssigmas:
        _, _, R_true_sqrt, _ = sym_sqrt_and_invsqrt((ssigma ** 2) * C_true @ C_true.T)
        ys_full_cache[ssigma] = ys_clean + vmap(lambda e: (R_true_sqrt @ e.T).T)(base_noise)

    # Output csv: one row per (ssigma, n_obs) with the mean, std and median over trials of each metric
    metric_names = ["KL_qstar_q", "KL_q_qstar", "NELBO", "kl_loss", "rec_loss", "prior_loss"]
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, output_name + ".csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["ssigma", "n_obs", "lr"] + [name + suffix for name in metric_names for suffix in ("", "_std", "_median")])

    n_settings = len(ssigmas) * len(n_obs_grid)
    for setting_idx, (ssigma, n_obs) in enumerate(itertools.product(ssigmas, n_obs_grid)):
        logger.info(f"Starting setting {setting_idx + 1}/{n_settings}: ssigma={ssigma}, n_obs={n_obs}")
        output_params_true = {"C": C_true, "d": d_true, "R": (ssigma ** 2) * C_true @ C_true.T}
        idx_obs = obs_index_cache[n_obs]
        obs_times = t_grid[idx_obs]
        ys_obs = ys_full_cache[ssigma][jnp.arange(n_trials)[:, None], idx_obs, :]

        # Initialize the posterior from the observations projected to the latent space
        if posterior_type == "grid":
            z_hat = vmap(vmap(lambda y: C_inv @ (y - d_true)))(ys_obs) # (B, n_obs, K)
            m_grid_init = jnp.mean(vmap(lambda ot, z: jnp.stack([jnp.interp(grid_times, ot, z[:, k]) for k in range(K)], axis=-1))(obs_times, z_hat), axis=0) # interpolate each trial onto the grid and average over trials
            post_var = 1.0 / (2 * alpha + 1.0 / ssigma ** 2) # initial posterior variance, giving R = sqrt(post_var) I via make_cholesky
            raw_R_init = jnp.zeros((grid_size + 1, K * (K + 1) // 2)).at[:, :K].set(inverse_softplus(jnp.sqrt(post_var)))
            post_net = GridInferenceNetwork(K=K, grid_times=grid_times, m_grid_init=m_grid_init, raw_R_grid_init=raw_R_init)
        elif posterior_type == "gp":
            z_hat = vmap(lambda y: C_inv @ (y - d_true))(ys_obs.reshape(-1, K))
            gp_init = init_gp_posterior(n_tau, K, (0.0, t_max), train_t=obs_times.reshape(-1), train_y=z_hat, len_init=1.0, full_cov=True)
            post_net = GPInferenceNetwork(K=K, init_params=gp_init, full_cov=True)

        training_key, key_train = jr.split(training_key, 2)
        key_train, eval_key_base = jr.split(key_train, 2)

        # Train
        params, _ = train(
            key=key_train,
            ys=ys_obs,
            obs_times=obs_times,
            likelihood=likelihood,
            output_params=output_params_true,
            t_max=t_max,
            encoder=encoder,
            process_ctx=process_ctx,
            post_net=post_net,
            prior=sde_true,
            sde_params=sde_params_true,
            init_params=init_params_true,
            learning_rate_init=lr,
            learning_rate_final=lr_final,
            mc_samples=mc_samples,
            jitter=1e-8,
            n_iters=n_iters,
            div_free=div_free,
            grad_clip_norm=5.0,
            learn_prior=False,
            learn_output=False,
            gauge=gauge,
            disable_pbar=True,
            process_data=partial(process_data_full_kl, t_max=t_max),
            per_trial_posterior=True,
            param_map=param_map,
        )
        params = param_map(params) if param_map is not None else params # broadcast the shared kernel hyperparameters to every trial

        # Evaluate every trial on its own
        results = {name: [] for name in metric_names}
        for trial in range(n_trials):
            logger.info(f"Evaluating trial {trial + 1}/{n_trials}")
            ys_trial, obs_times_trial = ys_obs[trial:trial + 1], obs_times[trial:trial + 1]
            params_trial = {**params, "posterior_params": jax.tree.map(lambda x: x[trial], params["posterior_params"])}
            _, eval_key_elbo = jr.split(jr.fold_in(eval_key_base, trial), 2)

            kl_qstar_q, kl_q_qstar = eval_posterior(sde_params=sde_params_true, init_params=init_params_true, ys=ys_trial, obs_times=obs_times_trial, t_grid=t_grid, encoder=encoder, post_net=post_net, params=params_trial, output_params=output_params_true, process_ctx=process_ctx, prior=sde_true, gauge=gauge, div_free=div_free_eval, learn_output=False)
            nelbo, kl_term, rec_term, prior_term = batched_eval_elbo(eval_key=eval_key_elbo, ys_obs=ys_trial, obs_times=obs_times_trial, encoder=encoder, post_net=post_net, params=params_trial, prior=sde_true, likelihood=likelihood, process_ctx=process_ctx, t_max=t_max, gauge=gauge, div_free=div_free_eval, batch_size=1)
            for name, value in zip(metric_names, (kl_qstar_q, kl_q_qstar, nelbo, kl_term, rec_term, prior_term)):
                results[name].append(value.item())
            logger.info(f"Finished trial {trial + 1}/{n_trials}: " + ", ".join(f"{name}={results[name][-1]:.6g}" for name in metric_names))

        logger.info(f"Finished setting {setting_idx + 1}/{n_settings}: " + ", ".join(f"{name}={jnp.mean(jnp.array(results[name])):.6g}" for name in metric_names))
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([ssigma, n_obs, lr] + [stat(jnp.array(results[name])).item() for name in metric_names for stat in (jnp.mean, jnp.std, jnp.median)])

if __name__ == "__main__":
    main()
