"""
Trains a latent SDE on the top K POD coefficients of the 2D cylinder flow
"""

import argparse
import logging
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import flax.linen as nn

jax.config.update("jax_enable_x64", True)

from helmholtz_sde.data import process_data_identity
from helmholtz_sde.likelihood import Gaussian
from helmholtz_sde.posterior.encoder import NullEncoder, constant_ctx
from helmholtz_sde.posterior.kernel_glm_posterior import GPInferenceNetwork, init_gp_posterior
from helmholtz_sde.sde import NeuralSDE, DriftNetwork
from helmholtz_sde.train import train
from helmholtz_sde.utils.general_helpers import inverse_softplus

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) # shared experiment code
from experiment_utils import build_correction

logger = logging.getLogger(__name__)


# --------------------- Helpers ---------------------
def constant_diagonal_diffusion(params: jnp.array, x: jnp.array, t: jnp.array) -> jnp.array:
    """
    Learned state-independent diagonal diffusion coefficient diag(softplus(params))
    """
    return jnp.diag(jax.nn.softplus(params))


def method_name(gauge: str, div_free: str) -> str:
    """
    Method under which the checkpoint is filed: sde_matching or svise without a correction, otherwise the correction
    """
    if div_free == "none":
        return "sde_matching" if gauge == "sqrt" else "svise"
    return div_free


# --------------------- Main function ---------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for name in ("jax", "jax._src.xla_bridge", "jaxlib"): # silence JAX backend warnings
        logging.getLogger(name).setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="2D cylinder flow experiment")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")

    # Observations
    parser.add_argument("--subsample_every", type=int, default=3)
    parser.add_argument("--obs_noise", type=float, default=0.3)
    parser.add_argument("--noise_mode", type=str, default="proportional", choices=["proportional", "constant"])

    # Model and training
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--n_tau", type=int, default=500)
    parser.add_argument("--gauge", type=str, choices=["sqrt", "sym"], default="sqrt")
    parser.add_argument("--div_free", type=str, default="none", choices=["none", "taylor", "least_squares"])
    parser.add_argument("--ell", type=int, default=1)
    parser.add_argument("--kappa", type=int, default=1)
    parser.add_argument("--n_mc", type=int, default=1)
    parser.add_argument("--mc_samples", type=int, default=512)
    parser.add_argument("--lr_init", type=float, default=1e-3)
    parser.add_argument("--lr_final", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=None)
    parser.add_argument("--n_iters", type=int, default=30000)
    parser.add_argument("--key_init", type=int, default=0)
    args = parser.parse_args()

    # Parse arguments
    (
        dataset_dir,
        ckpt_dir,
        subsample_every,
        obs_noise,
        noise_mode,
        hidden_size,
        depth,
        n_tau,
        gauge,
        div_free_name,
        ell,
        kappa,
        n_mc,
        mc_samples,
        lr_init,
        lr_final,
        grad_clip_norm,
        n_iters,
        key_init,
    ) = (
        args.dataset_dir,
        args.ckpt_dir,
        args.subsample_every,
        args.obs_noise,
        args.noise_mode,
        args.hidden_size,
        args.depth,
        args.n_tau,
        args.gauge,
        args.div_free,
        args.ell,
        args.kappa,
        args.n_mc,
        args.mc_samples,
        args.lr_init,
        args.lr_final,
        args.grad_clip_norm,
        args.n_iters,
        args.key_init,
    )
    div_free = build_correction(div_free_name, ell, kappa, n_mc)
    method = method_name(gauge, div_free_name)
    logger.info(f"Method = {method}, gauge = {gauge}, Helmholtz correction = {div_free}")

    # Dataset: the POD coefficients of the training frames, subsampled in time and observed with Gaussian noise
    with open(os.path.join(dataset_dir, "encoded_data.pkl"), "rb") as f:
        blob = pickle.load(f)
    z_train = np.array(blob["z_train"]) # (T_full, K)
    t_train = np.array(blob["t_train"])
    t_train = t_train - t_train[0]
    z_obs, obs_times = z_train[::subsample_every], t_train[::subsample_every]
    rng = np.random.default_rng(42)
    if obs_noise > 0:
        code_stdev = obs_noise * z_train.std(axis=0) if noise_mode == "proportional" else obs_noise # noise std proportional to the std of each mode
        z_obs = z_obs + code_stdev * rng.standard_normal(z_obs.shape)
    else:
        code_stdev = blob["code_stdev"]
    ys = jnp.array(z_obs)[None] # (B, T, D), a single trial
    obs_times = jnp.array(obs_times)[None] # (B, T)
    T, K = ys.shape[1], ys.shape[2] # the latent dimension equals the number of POD coefficients
    t_max = obs_times.max().item()
    logger.info(f"Loaded {dataset_dir}: {T} observations of {K} POD coefficients, every {subsample_every} frames on [0, {t_max:.1f}], noise {obs_noise} ({noise_mode})")

    # Posterior: one GP per trial (kernel GLM with a spectral covariance), no amortization
    encoder, process_ctx = NullEncoder(value=0.), constant_ctx
    full_cov = True
    gp_init = init_gp_posterior(n_tau, K, (0.0, t_max), obs_times[0], ys[0], len_init=None, full_cov=full_cov, key=jr.PRNGKey(key_init + 1)) # the lengthscale is cross-validated
    len_init = float(jax.nn.softplus(gp_init["mean"]["raw_len"])) # the selected lengthscale, stored so that the posterior can be rebuilt without the cross-validation
    gp_init = {**gp_init, "eigenvals": {**gp_init["eigenvals"], "b": inverse_softplus(jnp.var(ys[0], axis=0))}} # posterior variance of each mode is initialized to the variance of the noisy observations
    post_net = GPInferenceNetwork(K=K, init_params=gp_init, full_cov=full_cov)

    # Prior: drift network with tanh activations and a learned constant diagonal diffusion coefficient
    # NOTE: we observed that a learnable, state-dependent diffusion coefficient worsened performance for SDE Matching (Bartosh et al., 2025)
    drift_net = DriftNetwork(hidden_dim=hidden_size, K=K, depth=depth, activation=nn.tanh)
    sde_params = {"network_params_drift": drift_net.init(jr.PRNGKey(key_init + 2), jnp.zeros((K,)), jnp.zeros((1,))), "network_params_diffusion": inverse_softplus(jnp.ones(K))}
    prior = NeuralSDE(K, apply_fn_drift=drift_net.apply, apply_fn_diffusion=constant_diagonal_diffusion)
    init_params = {"mu0": jnp.zeros((K,)), "V0": jnp.eye(K)}

    # Likelihood with the observation noise fixed to its true variance
    output_params = {"C": jnp.eye(K), "d": jnp.zeros((K,)), "R": jnp.ones(K) * jnp.asarray(code_stdev) ** 2}
    likelihood = Gaussian()

    params, metrics = train(
        jr.PRNGKey(key_init + 4),
        ys,
        obs_times,
        likelihood,
        output_params,
        t_max,
        encoder,
        process_ctx,
        post_net,
        prior,
        sde_params,
        init_params,
        learning_rate_init=lr_init,
        learning_rate_final=lr_final,
        mc_samples=mc_samples,
        jitter=1e-8,
        n_iters=n_iters,
        div_free=div_free,
        learn_prior=True,
        learn_output=False,
        gauge=gauge,
        grad_clip_norm=grad_clip_norm,
        process_data=process_data_identity,
        per_trial_posterior=True,
        disable_pbar=True,
    )
    logger.info(f"Finished {n_iters} iterations: final loss {metrics['loss'][-1]:.4f}, mean loss over the last 1000 iterations {jnp.mean(metrics['loss'][-1000:]):.4f}")

    # Save the checkpoint under the method
    ckpt_basename = f"sub{subsample_every}_noise{obs_noise}_fullcov{full_cov}_gauge{gauge}_h{hidden_size}_d{depth}"
    if div_free_name == "taylor":
        ckpt_basename += f"_ell{ell}"
    elif div_free_name == "least_squares":
        ckpt_basename += f"_ell{ell}_k{kappa}_n{n_mc}"
    ckpt_path = os.path.join(ckpt_dir, method, ckpt_basename + f"_key{key_init}.pkl")
    config = {
        "subsample_every": subsample_every,
        "obs_noise": obs_noise,
        "noise_mode": noise_mode,
        "code_stdev": code_stdev,
        "K": K,
        "T": T,
        "t_max": t_max,
        "n_tau": n_tau,
        "len_init": len_init,
        "full_cov": full_cov,
        "hidden_size": hidden_size,
        "depth": depth,
        "gauge": gauge,
        "div_free": div_free_name,
        "ell": ell,
        "kappa": kappa,
        "n_mc": n_mc,
        "mc_samples": mc_samples,
        "lr_init": lr_init,
        "lr_final": lr_final,
        "grad_clip_norm": grad_clip_norm,
        "n_iters": n_iters,
        "key_init": key_init,
    }
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    with open(ckpt_path, "wb") as f:
        pickle.dump({"params": params, "metrics": metrics, "config": config}, f)
    logger.info(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
