"""
Reproduces the Ornstein-Uhlenbeck spiral learning experiment from the paper (learned prior and output model)
"""

import argparse
import csv
import itertools
import logging
import os
import pickle
import sys
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap

jax.config.update("jax_enable_x64", True)

from helmholtz_sde.sde import SDE, LinearSDE, NeuralSDE, DriftNetwork
from helmholtz_sde.likelihood import Gaussian
from helmholtz_sde.posterior.posterior import Posterior
from helmholtz_sde.posterior.encoder import weighted_ctx, constant_ctx, NullEncoder, ForwardGRUEncoder
from helmholtz_sde.posterior.nn_posterior import InferenceNetwork
from helmholtz_sde.posterior.kernel_glm_posterior import GPInferenceNetwork, init_gp_posterior, share_kernel_hyperparameters
from helmholtz_sde.posterior.drift import apply_inference_net_time_derivs, get_reference_drift_fn
from helmholtz_sde.train import train
from helmholtz_sde.data import process_data_full_kl
from helmholtz_sde.utils.general_helpers import gaussian_kl, simulate_learned_prior_samples, get_transformation_for_latents, transform_marginals, transform_vector_field, sym_sqrt_and_invsqrt
from helmholtz_sde.helmholtz.correction import HelmCorrection

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) # shared experiment code
from experiment_utils import build_correction, eval_correction, batched_mean_eval, batched_eval_elbo
from spiral_utils import run_sing_smoother

logger = logging.getLogger(__name__)


# --------------------- Helpers ---------------------
def _latent_transformation(output_params_true: Dict[str, jnp.array], output_params: Dict[str, jnp.array], learn_output: bool) -> Optional[Tuple[jnp.array, jnp.array]]:
    """
    Affine map x = P x' + offset from the learned to the true latent coordinates
    """
    K = output_params_true["C"].shape[1]
    P, offset = get_transformation_for_latents(C=output_params_true["C"], d=output_params_true["d"], C_hat=output_params["C"], d_hat=output_params["d"], Sigma=jnp.eye(K))
    if not learn_output and not (jnp.allclose(P, jnp.eye(K), atol=1e-6) and jnp.allclose(offset, 0.0, atol=1e-6)):
        raise ValueError("the latent transformation must be the identity when the output parameters are fixed")
    if not jnp.allclose(P.T @ P, jnp.eye(K), atol=1e-6): # tolerance well above the 1e-8 jitter of the matrix square roots
        logger.warning(f"Latent transformation is not orthogonal (max |P^T P - I| = {jnp.max(jnp.abs(P.T @ P - jnp.eye(K))):.1e}); the KL divergences are undefined")
        return None
    return P, offset


def _expected_sq_norm_quadratic(C: jnp.array, A: jnp.array, b: jnp.array, m: jnp.array, S: jnp.array) -> jnp.array:
    """
    E||q(x) + A x + b||^2 under x ~ N(m, S), where q_i(x) = x^T C_i x for the (K, K, K) tensor C
    """
    C = 0.5 * (C + jnp.swapaxes(C, 1, 2)) # x^T C_i x = x^T sym(C_i) x
    Eq = jnp.einsum("iab,ab->i", C, S) + jnp.einsum("iab,a,b->i", C, m, m) # (K), E[q(x)]

    # Linear term E||A x + b||^2
    mu_lin = A @ m + b
    E_lin_norm2 = jnp.dot(mu_lin, mu_lin) + jnp.trace(A @ S @ A.T)

    # Cross term 2 E[q(x)^T (A x + b)], with E[(x^T C_i x) x] = E[q_i] m + 2 S C_i m
    SC_m = jnp.einsum("ab,ibc,c->ia", S, C, m)
    E_x_q = m[None, :] * Eq[:, None] + 2.0 * SC_m # (K, K), row i is E[(x^T C_i x) x]^T
    cross = 2.0 * (jnp.einsum("ia,ia->", E_x_q, A) + jnp.dot(Eq, b))

    # Quadratic term E||q(x)||^2, with E[(x^T C_i x)^2] = (tr(C_i S) + m^T C_i m)^2 + 2 tr((C_i S)^2) + 4 m^T C_i S C_i m
    CS = jnp.einsum("iab,bc->iac", C, S)
    tr_CS2 = jnp.einsum("iab,iba->i", CS, CS)
    C_m = jnp.einsum("iab,b->ia", C, m) # (K, K)
    m_C_S_C_m = jnp.einsum("ia,ab,ib->i", C_m, S, C_m) # (K)
    E_q_norm2 = jnp.sum(Eq ** 2 + 2.0 * tr_CS2 + 4.0 * m_C_S_C_m)
    return E_q_norm2 + E_lin_norm2 + cross


# --------------------- Evaluation functions ---------------------
def eval_prior_drift(key: jr.PRNGKey, sde: SDE, sde_params: Dict[str, jnp.array], init_params: Dict[str, jnp.array], xs_test: jnp.array, t_grid: jnp.array, prior: SDE, params: Dict[str, Any], output_params: Dict[str, jnp.array], learn_output: bool = False) -> Tuple[jnp.array, jnp.array]:
    """
    KL(p* || p) and KL(p || p*) between the true prior p* and the learned prior p, from the KL between the initial
    distributions and the drift mismatch along sample paths of each (xs_test from p*, fresh samples from p)
    
    NOTE: assumes the diffusion coefficient of both priors is the identity
    """
    transformation = _latent_transformation(output_params, params["output_params"], learn_output)
    if transformation is None:
        return jnp.inf, jnp.inf
    P, offset = transformation
    dt = t_grid[1:] - t_grid[:-1]

    f_true = lambda x: sde.drift(x, jnp.array([0.]), sde_params)
    f_learned = transform_vector_field(partial(prior.drift, t=jnp.array([0.]), sde_params=params["sde_params"]), P, offset) # learned drift in the true coordinates
    mu0, V0 = P @ params["init_params"]["mu0"] + offset, P @ params["init_params"]["V0"] @ P.T # learned initial distribution in the true coordinates

    def _path_kl(xs: jnp.array) -> jnp.array:
        """
        Integral of (1/2)||f_true - f_learned||^2 along one path, the KL rate for the identity diffusion coefficient
        """
        diff = vmap(f_true)(xs[:-1]) - vmap(f_learned)(xs[:-1]) # (T - 1, K)
        return 0.5 * jnp.sum(dt * jnp.sum(jnp.square(diff), axis=-1))

    kl_pstar_p = gaussian_kl(init_params["mu0"], init_params["V0"], mu0, V0) + jnp.mean(vmap(_path_kl)(xs_test))
    xs_prior = simulate_learned_prior_samples(key, prior, params, t_max=t_grid[-1], n_timesteps=t_grid.shape[0] - 1, n_eval=xs_test.shape[0]) # in the learned coordinates
    xs_prior = vmap(vmap(lambda x: P @ x + offset))(xs_prior)
    kl_p_pstar = gaussian_kl(mu0, V0, init_params["mu0"], init_params["V0"]) + jnp.mean(vmap(_path_kl)(xs_prior))
    return kl_pstar_p, kl_p_pstar


def eval_posterior(sde_params: Dict[str, jnp.array], init_params: Dict[str, jnp.array], ys: jnp.array, obs_times: jnp.array, t_grid: jnp.array, encoder: Union[NullEncoder, ForwardGRUEncoder], post_net: Posterior, params: Dict[str, Any], output_params: Dict[str, jnp.array], process_ctx: Callable, prior: SDE, gauge: str = "sqrt", div_free: Optional[HelmCorrection] = None, learn_output: bool = False, per_trial_posterior: bool = False) -> Tuple[jnp.array, jnp.array]:
    """
    KL(q* || q) and KL(q || q*) between the exact posterior q* of the linear-Gaussian model (sde_params, init_params,
    output_params) and the learned posterior q, averaged over trials
    
    NOTE: assumes the diffusion coefficient of both models is the identity
    """
    if div_free is not None and div_free.ell > 2:
        raise ValueError(f"eval_posterior supports Helmholtz corrections of degree at most 2, got {div_free}")
    transformation = _latent_transformation(output_params, params["output_params"], learn_output)
    if transformation is None:
        return jnp.inf, jnp.inf
    P, offset = transformation
    K = prior.K
    I = jnp.eye(K)
    del_t = t_grid[1:] - t_grid[:-1]

    # Exact posterior on the grid
    ms, Ss, SSs = run_sing_smoother(ys, obs_times, t_grid, sde_params, init_params, output_params) # (B, T, K), (B, T, K, K), (B, T - 1, K, K)

    def _transitions(ms: jnp.array, Ss: jnp.array, SSs: jnp.array) -> Tuple[jnp.array, jnp.array]:
        As = vmap(lambda S, SS: jnp.linalg.solve(S, SS.T).T)(Ss[:-1], SSs) # (T - 1, K, K)
        bs = vmap(lambda m, m_next, A: m_next - A @ m)(ms[:-1], ms[1:], As) # (T - 1, K)
        return (As - I) / del_t[:, None, None], bs / del_t[:, None]

    As, bs = vmap(_transitions)(ms, Ss, SSs) # (B, T - 1, K, K), (B, T - 1, K)

    # Learned posterior on the grid
    def _learned_posterior_at(ctx_seq: jnp.array, proc_ctx: Callable, post_params: Dict[str, Any], t: jnp.array) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]:
        t_arr = jnp.array([t], dtype=ctx_seq.dtype)
        mt, Rt, dmt, dRt = apply_inference_net_time_derivs(post_net, post_params, t_arr, ctx_seq, proc_ctx)
        fq = get_reference_drift_fn(gauge)(mt, Rt, dmt, dRt, I) # affine in x
        zero = jnp.zeros((K,), dtype=mt.dtype)
        Ct, At, bt = jnp.zeros((K, K, K), dtype=mt.dtype), jax.jacfwd(fq)(zero), fq(zero)
        if div_free is not None:
            fp = lambda x: prior.drift(x, t_arr, params["sde_params"])
            terms = div_free.monomial_coeffs(div_free.fit(mt, Rt @ Rt.T, fp, fq)) # highest degree first
            while len(terms) < 3: # pad to (quadratic, linear, constant)
                terms = (jnp.zeros((K,) * (len(terms) + 1), dtype=mt.dtype),) + terms
            Ct, At, bt = Ct + terms[0], At + terms[1], bt + terms[2]
        return mt, Rt @ Rt.T, Ct, At, bt

    def _trial(y: jnp.array, obs_t: jnp.array, post_params: Dict[str, Any]):
        ctx_seq = encoder.apply(params["encoder_params"], y)
        return vmap(partial(_learned_posterior_at, ctx_seq, partial(process_ctx, obs_t), post_params))(t_grid[:-1])

    if per_trial_posterior: # the posterior parameters are sliced along with the trial
        _, (ms_approx, Ss_approx, Cs_approx, As_approx, bs_approx) = lax.scan(lambda _, inputs: (None, _trial(*inputs)), None, (ys, obs_times, params["posterior_params"]))
    else:
        _, (ms_approx, Ss_approx, Cs_approx, As_approx, bs_approx) = lax.scan(lambda _, inputs: (None, _trial(*inputs, params["posterior_params"])), None, (ys, obs_times))

    if learn_output: # express the learned posterior in the true latent coordinates x = P x' + offset
        Q, s = P.T, P.T @ offset
        ms_approx, Ss_approx = transform_marginals(ms_approx, Ss_approx, P, offset)

        def _transform_drift(C: jnp.array, A: jnp.array, b: jnp.array) -> Tuple[jnp.array, jnp.array, jnp.array]:
            """
            Coefficients of x -> P f(Q x - s) for f(x') = C[x', x'] + A x' + b; the quadratic term feeds all three orders
            """
            C_new = jnp.einsum("ri,ijk,ja,kb->rab", P, C, Q, Q)
            A_new = P @ A @ Q + (-jnp.einsum("ri,ijk,ja,k->ra", P, C, Q, s) - jnp.einsum("ri,ijk,j,ka->ra", P, C, s, Q))
            b_new = P @ (b - A @ s) + jnp.einsum("ri,ijk,j,k->r", P, C, s, s)
            return C_new, A_new, b_new

        Cs_approx, As_approx, bs_approx = vmap(vmap(_transform_drift))(Cs_approx, As_approx, bs_approx)

    # Compute E||f*(x) - f(x)||^2 under each posterior, with f*(x) - f(x) = -C[x, x] + (A* - A) x + (b* - b)
    Cs, As, bs = -Cs_approx, As - As_approx, bs - bs_approx
    rates_qstar_q = vmap(vmap(_expected_sq_norm_quadratic))(Cs, As, bs, ms[:, :-1], Ss[:, :-1]) # (B, T - 1)
    rates_q_qstar = vmap(vmap(_expected_sq_norm_quadratic))(Cs, As, bs, ms_approx, Ss_approx) # (B, T - 1)
    kl_qstar_q = jnp.mean(vmap(gaussian_kl)(ms[:, 0], Ss[:, 0], ms_approx[:, 0], Ss_approx[:, 0]) + 0.5 * jnp.sum(del_t * rates_qstar_q, axis=-1))
    kl_q_qstar = jnp.mean(vmap(gaussian_kl)(ms_approx[:, 0], Ss_approx[:, 0], ms[:, 0], Ss[:, 0]) + 0.5 * jnp.sum(del_t * rates_q_qstar, axis=-1))
    return kl_qstar_q, kl_q_qstar


# --------------------- Main function ---------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for name in ("jax", "jax._src.xla_bridge", "jaxlib"): # silence JAX backend warnings
        logging.getLogger(name).setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Ornstein-Uhlenbeck spiral learning experiment")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default="spiral_exp")

    # Observations
    parser.add_argument("--n_trials", type=int, default=1024)
    parser.add_argument("--ssigmas", type=float, nargs="+", default=[0.05, 0.3, 1.0])
    parser.add_argument("--n_obs_grid", type=int, nargs="+", default=[5, 10, 20, 50])

    # Model and training
    parser.add_argument("--alpha0", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--mc_samples", type=int, default=16)
    parser.add_argument("--n_iters", type=int, default=30000)
    parser.add_argument("--n_reps", type=int, default=5)
    parser.add_argument("--gauge", type=str, choices=["sqrt", "sym"], default="sqrt")
    parser.add_argument("--div_free", type=str, default="none", choices=["none", "taylor", "least_squares"])
    parser.add_argument("--ell", type=int, default=1)
    parser.add_argument("--kappa", type=int, default=1)
    parser.add_argument("--n_mc", type=int, default=1)
    parser.add_argument("--n_nodes", type=int, default=None)
    parser.add_argument("--learn_prior", type=int, choices=[0, 1], default=1)
    parser.add_argument("--learn_output", type=int, choices=[0, 1], default=1)
    parser.add_argument("--per_trial_posterior", type=int, choices=[0, 1], default=0)
    parser.add_argument("--posterior_type", type=str, default="gp", choices=["gp", "nn"])
    parser.add_argument("--post_hidden_dim", type=int, default=100)
    parser.add_argument("--post_depth", type=int, default=2)
    parser.add_argument("--n_tau", type=int, default=200)

    # Evaluation
    parser.add_argument("--n_nodes_eval", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    args = parser.parse_args()

    # Parse arguments
    (
        dataset_dir,
        output_dir,
        output_name,
        n_trials,
        ssigmas,
        n_obs_grid,
        alpha0,
        lr,
        hidden_dim,
        embedding_size,
        depth,
        mc_samples,
        n_iters,
        n_reps,
        gauge,
        div_free,
        ell,
        kappa,
        n_mc,
        n_nodes,
        learn_prior,
        learn_output,
        per_trial_posterior,
        posterior_type,
        post_hidden_dim,
        post_depth,
        n_tau,
        n_nodes_eval,
        eval_batch_size,
    ) = (
        args.dataset_dir,
        args.output_dir,
        args.output_name,
        args.n_trials,
        args.ssigmas,
        args.n_obs_grid,
        args.alpha0,
        args.lr,
        args.hidden_dim,
        args.embedding_size,
        args.depth,
        args.mc_samples,
        args.n_iters,
        args.n_reps,
        args.gauge,
        args.div_free,
        args.ell,
        args.kappa,
        args.n_mc,
        args.n_nodes,
        args.learn_prior,
        args.learn_output,
        args.per_trial_posterior,
        args.posterior_type,
        args.post_hidden_dim,
        args.post_depth,
        args.n_tau,
        args.n_nodes_eval,
        args.eval_batch_size,
    )
    learn_prior, learn_output, per_trial_posterior = bool(learn_prior), bool(learn_output), bool(per_trial_posterior)
    div_free = build_correction(div_free, ell, kappa, n_mc, n_nodes)
    if div_free is not None and div_free.ell > 2:
        raise ValueError(f"the KL to the exact posterior supports Helmholtz corrections of degree at most 2, got --ell {ell}")
    div_free_eval = eval_correction(div_free, n_nodes=n_nodes_eval)
    per_trial_posterior = per_trial_posterior or posterior_type == "gp" # the GP posterior is always fit per trial
    logger.info(f"Learning prior = {learn_prior}, learning output = {learn_output}, gauge = {gauge}, Helmholtz correction = {div_free} (evaluated with {div_free_eval})")

    # Dataset
    with open(os.path.join(dataset_dir, "train.pkl"), "rb") as f:
        blob_train = pickle.load(f)
    with open(os.path.join(dataset_dir, "test.pkl"), "rb") as f:
        xs_test = jnp.array(pickle.load(f))
    xs_train = jnp.array(blob_train["xs"])
    t_grid = jnp.array(blob_train["t_grid"])
    alpha, omega = float(blob_train["sde_params"]["alpha"]), float(blob_train["sde_params"]["kappa"]) # decay coefficient and rotation rate (stored under the key kappa)
    C_true, d_true = jnp.array(blob_train["output_params"]["C"]), jnp.array(blob_train["output_params"]["d"])
    K = xs_train.shape[-1]
    n_test, n_timesteps, t_max = xs_test.shape[0], t_grid.shape[0] - 1, t_grid[-1].item()
    if n_trials > xs_train.shape[0]:
        raise ValueError(f"Requested n_trials={n_trials}, but the dataset only has {xs_train.shape[0]} training trials")
    xs_train = xs_train[:n_trials]
    logger.info(f"Loaded dataset from {dataset_dir}; using the first {n_trials} training trials and {n_test} test trials")

    # True generative model
    J = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    sde_true = LinearSDE(K)
    sde_params_true = {"A": -alpha * jnp.eye(K) + omega * J, "b": jnp.zeros((K))}
    init_params_true = {"mu0": jnp.zeros((K)), "V0": (1.0 / (2.0 * alpha)) * jnp.eye(K)} # stationary distribution
    likelihood = Gaussian()

    # Posterior
    param_map = None
    if posterior_type == "gp": # per-trial GP posterior with kernel hyperparameters shared across trials
        encoder, process_ctx_base = NullEncoder(value=0.), constant_ctx
        gp_init = init_gp_posterior(n_tau, K, (0.0, t_max), train_t=jnp.linspace(0.0, t_max, 50), train_y=jnp.zeros((50, K)), len_init=1.0, full_cov=True) # the mean is initialized at zero since the output model is unknown
        post_net = GPInferenceNetwork(K=K, init_params=gp_init, full_cov=True)
        param_map = share_kernel_hyperparameters
        logger.info(f"GP posterior with n_tau={n_tau}, shared kernel hyperparameters")
    elif per_trial_posterior:
        encoder, process_ctx_base = NullEncoder(value=0.), constant_ctx
        post_net = InferenceNetwork(hidden_dim=post_hidden_dim, K=K, depth=post_depth, jitter=1e-8)
    else:
        encoder, process_ctx_base = ForwardGRUEncoder(embedding_size), weighted_ctx
        post_net = InferenceNetwork(hidden_dim=hidden_dim, K=K, depth=depth, jitter=1e-8)

    # Noise-free observations of the training trials, observation times for each n_obs and observation noise for each ssigma
    ys_clean = vmap(vmap(lambda x: C_true @ x + d_true))(xs_train) # (B, T, D)
    obs_index_cache = {}
    for n_obs in n_obs_grid:
        idx_key = jr.fold_in(jr.PRNGKey(1), n_obs)
        obs_index_cache[n_obs] = vmap(lambda key: jnp.sort(jr.choice(key, a=n_timesteps + 1, shape=(n_obs,), replace=False)))(jr.split(idx_key, n_trials)) # (B, n_obs)
    ys_full_cache = {}
    for ssigma, eps_key in zip(ssigmas, jr.split(jr.PRNGKey(2), len(ssigmas))):
        _, _, R_true_sqrt, _ = sym_sqrt_and_invsqrt((ssigma ** 2) * (C_true @ C_true.T))
        eps = jr.normal(eps_key, shape=ys_clean.shape, dtype=ys_clean.dtype)
        ys_full_cache[ssigma] = ys_clean + vmap(lambda e: (R_true_sqrt @ e.T).T)(eps)

    # Output csv: one row per (ssigma, n_obs) with the mean, std and median over replicates of each metric
    metric_names = ["KL_pstar_p", "KL_p_pstar", "KL_qstar_q", "KL_q_qstar", "NELBO", "kl_loss", "rec_loss", "prior_loss"]
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, output_name + ".csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["ssigma", "n_obs", "lr"] + [name + suffix for name in metric_names for suffix in ("", "_std", "_median")])

    training_key = jr.PRNGKey(3)
    n_settings = len(ssigmas) * len(n_obs_grid)
    for setting_idx, (ssigma, n_obs) in enumerate(itertools.product(ssigmas, n_obs_grid)):
        logger.info(f"Starting setting {setting_idx + 1}/{n_settings}: ssigma={ssigma}, n_obs={n_obs}")
        output_params_true = {"C": C_true, "d": d_true, "R": (ssigma ** 2) * (C_true @ C_true.T)}
        idx_obs = obs_index_cache[n_obs]
        obs_times = t_grid[idx_obs]
        ys_obs = ys_full_cache[ssigma][jnp.arange(n_trials)[:, None], idx_obs, :]
        beta = ((n_obs - 1) ** 2) / (alpha0 * 2.0 * t_max ** 2) # the weighted context depends on n_obs, so it is rebound for each setting
        process_ctx = partial(weighted_ctx, beta=beta) if process_ctx_base is weighted_ctx else process_ctx_base

        results = {name: [] for name in metric_names}
        training_key, setting_key = jr.split(training_key, 2)
        for rep in range(n_reps):
            logger.info(f"Rep {rep + 1}/{n_reps} for ssigma={ssigma}, n_obs={n_obs}")
            rep_key, eval_key, init_key = jr.split(jr.fold_in(setting_key, rep), 3)
            init_key_prior, init_key_output = jr.split(init_key, 2)

            # Prior and output parameters
            if learn_prior:
                drift_net = DriftNetwork(hidden_dim=hidden_dim, K=K, depth=1, include_time=False)
                sde_params = {"network_params_drift": drift_net.init(init_key_prior, jnp.zeros((K,)), jnp.zeros((1,)))} # the diffusion coefficient is fixed to the identity
                prior = NeuralSDE(K, apply_fn_drift=drift_net.apply)
                init_params = {"mu0": jnp.zeros((K,)), "V0": jnp.eye(K)}
            else:
                prior, sde_params, init_params = sde_true, sde_params_true, init_params_true
            if learn_output:
                C_init_key, d_init_key, R_init_key = jr.split(init_key_output, 3)
                output_params_init = {"C": jr.normal(key=C_init_key, shape=(K, K)), "d": jr.normal(key=d_init_key, shape=(K,)), "R": jnp.diag(jr.uniform(key=R_init_key, shape=(K,), minval=0.2, maxval=1.0))}
            else:
                output_params_init = output_params_true

            # Train
            params, _ = train(
                key=rep_key,
                ys=ys_obs,
                obs_times=obs_times,
                likelihood=likelihood,
                output_params=output_params_init,
                t_max=t_max,
                encoder=encoder,
                process_ctx=process_ctx,
                post_net=post_net,
                prior=prior,
                sde_params=sde_params,
                init_params=init_params,
                learning_rate_init=lr,
                learning_rate_final=lr,
                mc_samples=mc_samples,
                jitter=1e-8,
                n_iters=n_iters,
                div_free=div_free,
                learn_prior=learn_prior,
                learn_output=learn_output,
                gauge=gauge,
                disable_pbar=True,
                process_data=partial(process_data_full_kl, t_max=t_max),
                per_trial_posterior=per_trial_posterior,
                param_map=param_map,
            )
            params = param_map(params) if param_map is not None else params # broadcast the shared kernel hyperparameters to every trial

            # Evaluate the learned prior and the approximate posterior
            eval_key_prior, _, eval_key_elbo = jr.split(eval_key, 3)
            if learn_prior:
                kl_pstar_p, kl_p_pstar = batched_mean_eval(lambda i, j: eval_prior_drift(key=jr.fold_in(eval_key_prior, i), sde=sde_true, sde_params=sde_params_true, init_params=init_params_true, xs_test=xs_test[i:j], t_grid=t_grid, prior=prior, params=params, output_params=output_params_true, learn_output=learn_output), n_test, batch_size=eval_batch_size)
            else:
                kl_pstar_p, kl_p_pstar = jnp.array(jnp.nan), jnp.array(jnp.nan)
            kl_qstar_q, kl_q_qstar = batched_mean_eval(lambda i, j: eval_posterior(sde_params=sde_params_true, init_params=init_params_true, ys=ys_obs[i:j], obs_times=obs_times[i:j], t_grid=t_grid, encoder=encoder, post_net=post_net, params={**params, "posterior_params": jax.tree.map(lambda x: x[i:j], params["posterior_params"])} if per_trial_posterior else params, output_params=output_params_true, process_ctx=process_ctx, prior=prior, gauge=gauge, div_free=div_free_eval, learn_output=learn_output, per_trial_posterior=per_trial_posterior), n_trials, batch_size=eval_batch_size)
            nelbo, kl_term, rec_term, prior_term = batched_eval_elbo(eval_key=eval_key_elbo, ys_obs=ys_obs, obs_times=obs_times, encoder=encoder, post_net=post_net, params=params, prior=prior, likelihood=likelihood, process_ctx=process_ctx, t_max=t_max, gauge=gauge, div_free=div_free_eval, batch_size=eval_batch_size, per_trial_posterior=per_trial_posterior)
            for name, value in zip(metric_names, (kl_pstar_p, kl_p_pstar, kl_qstar_q, kl_q_qstar, nelbo, kl_term, rec_term, prior_term)):
                results[name].append(value.item())
            logger.info(f"Finished rep {rep + 1}/{n_reps}: " + ", ".join(f"{name}={results[name][-1]:.6g}" for name in metric_names))

        logger.info(f"Finished setting ssigma={ssigma}, n_obs={n_obs}: " + ", ".join(f"{name}={jnp.mean(jnp.array(results[name])):.6g}" for name in metric_names))
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([ssigma, n_obs, lr] + [stat(jnp.array(results[name])).item() for name in metric_names for stat in (jnp.mean, jnp.std, jnp.median)])


if __name__ == "__main__":
    main()
