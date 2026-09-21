import itertools
from functools import partial

import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap

from helmholtz_sde.sde import SDE
from helmholtz_sde.likelihood import Likelihood
from helmholtz_sde.posterior.posterior import Posterior
import tensorflow_probability.substrates.jax.distributions as tfd

from typing import Any, Callable, Dict, Optional, Tuple

# --------------------- SDE simulation ---------------------
def simulate_sde(
    key: jr.PRNGKey, # jax key for sampling
    x0: jnp.array, # initial value
    sde: SDE, # SDE from which to simulate
    sde_params: Dict[str, Any], # parameters of the SDE
    t_max: float = 1., # simulate the SDE on [0, t_max]
    n_timesteps: int = 1000, # number of integration steps
) -> jnp.array:
    """
    Simulate an SDE according to the Euler-Maruyama discretization
    """

    dt = t_max / n_timesteps

    keys = jr.split(key, n_timesteps)
    t_grid = jnp.linspace(0.0, t_max, n_timesteps + 1)[:-1] # (n_timesteps)

    def _step(x, arg):
        key_t, t = arg
        drift = sde.drift(x, t, sde_params)
        G = sde.diffusion(x, t, sde_params)
        Sigma = G @ G.T
        next_x = tfd.MultivariateNormalFullCovariance(
            loc = x + drift * dt,
            covariance_matrix = Sigma * dt
        ).sample(seed=key_t)
        return next_x, next_x # store the new state

    _, xs = lax.scan(_step, x0, (keys, t_grid))
    return jnp.concatenate([x0[None, :], xs], axis=0)


def simulate_learned_prior_samples(key: jr.PRNGKey, prior: SDE, params: Dict[str, Any], t_max: float, n_timesteps: int, n_eval: int) -> jnp.array:
    """
    Draw n_eval sample paths (n_eval, n_timesteps + 1, K) from the learned prior, with x(0) ~ N(mu0, V0)
    """
    mu0, V0 = params["init_params"]["mu0"], params["init_params"]["V0"]
    _, _, sqrtV0, _ = sym_sqrt_and_invsqrt(V0)
    key_init, key_paths = jr.split(key, 2)
    x0 = mu0[None, :] + (sqrtV0 @ jr.normal(key_init, shape=(n_eval, prior.K)).T).T
    return vmap(partial(simulate_sde, sde=prior, sde_params=params["sde_params"], t_max=t_max, n_timesteps=n_timesteps))(jr.split(key_paths, n_eval), x0)


# --------------------- Transforming subspaces ---------------------
def get_transformation_for_latents(
    C: jnp.array, # true output mapping
    d: jnp.array, # true offset vector
    C_hat: jnp.array, # learned output mapping
    d_hat: jnp.array, # learned offset vector
    Sigma: jnp.array, # diffusion matrix
    jitter: float = 1e-8 # jitter
) -> Tuple[jnp.array, jnp.array]:
    """
    Aligns the subspaces determined by (C, d) and (C_hat, d_hat) using Procrustes alignment
    x = Px' + offset (x' the inferred latent space, x the true latent space)
    where P is an (K, K) Σ-orthogonal matrix (P Σ P^T = Σ) and offset is a (K) vector
    """
    _, _, sqrtSigma, invsqrtSigma = sym_sqrt_and_invsqrt(Sigma, jitter=jitter)

    # Solve for P via a weighted Procrustes alignment
    A = C @ sqrtSigma
    B = C_hat @ sqrtSigma

    M = A.T @ B
    U1, _, U2t = jnp.linalg.svd(M, full_matrices=False)
    U = U1 @ U2t

    # Map back to original latent coordinates: P = sqrtSigma U invsqrtSigma
    P = sqrtSigma @ U @ invsqrtSigma

    # Offset chosen so that d_hat ≈ C offset + d
    offset = jnp.linalg.pinv(C) @ (d_hat - d)
    return P, offset


def transform_vector_field(
    f: Callable[[jnp.array], jnp.array], # vector field defined on the x' space, f(x') in R^K for x' in R^K
    P: jnp.array, # (K, K) linear part of the transformation
    offset: jnp.array # (K) additive part of the transformation
) -> Callable[[jnp.array], jnp.array]:
    """
    Transforms a vector field according to x = Px' + offset
    """
    def f_trans(x):
        x_inf = jnp.linalg.solve(P, x - offset)
        return P @ f(x_inf)
    return f_trans


def transform_marginals(ms: jnp.array, Ss: jnp.array, P: jnp.array, offset: jnp.array) -> Tuple[jnp.array, jnp.array]:
    """
    Transforms Gaussian marginals with means ms (..., K) and covariances Ss (..., K, K) according to x = Px' + offset
    """
    f_m = lambda m: P @ m + offset
    f_S = lambda S: P @ S @ P.T
    for _ in range(ms.ndim - 1):
        f_m, f_S = vmap(f_m), vmap(f_S)
    return f_m(ms), f_S(Ss)


# --------------------- Other ---------------------
def sym_sqrt_and_invsqrt(S: jnp.array, jitter: float = 1e-8) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    """
    Returns (Q, evals, sqrtS, invsqrtS) for psd matrix S using eigendecomposition
    """
    # NOTE: a symmetry-breaking perturbation is necessary: the backward pass of eigh computes
    # 1 / (lambda_i - lambda_j) which is NaN when eigenvalues are degenerate
    K = S.shape[0]
    evals, Q = jnp.linalg.eigh(S + jitter * jnp.diag(jnp.arange(1, K + 1, dtype=S.dtype)))
    evals = jnp.clip(evals, jitter)
    sqrt_e = jnp.sqrt(evals)
    invsqrt_e = 1.0 / sqrt_e
    sqrtS = (Q * sqrt_e[None, :]) @ Q.T
    invsqrtS = (Q * invsqrt_e[None, :]) @ Q.T
    return Q, evals, sqrtS, invsqrtS


def symmetrize_full(T: jnp.array, n_axes: int, start_axis: int = 0) -> jnp.array:
    """
    Fully symmetrize T over n_axes consecutive axes starting at start_axis
    """
    if n_axes <= 1:
        return T
    rank = T.ndim
    base = list(range(rank))
    leading = tuple(base[:start_axis])
    trailing = tuple(base[start_axis + n_axes:])
    sym_axes = base[start_axis:start_axis + n_axes]
    out = jnp.zeros_like(T)
    count = 0
    for perm in itertools.permutations(sym_axes):
        out = out + jnp.transpose(T, leading + tuple(perm) + trailing)
        count += 1
    return out / count


def inverse_softplus(x: float) -> jnp.array:
    x = jnp.array(x)
    return x + jnp.log(-jnp.expm1(-x))


# --------------------- KL computation ---------------------
def gaussian_kl(m0: jnp.array, S0: jnp.array, m1: jnp.array, S1: jnp.array, jitter: float = 1e-8) -> jnp.array:
    """
    Computes the Gaussian KL divergence KL(N(m0, S0) || N(m1, S1))
    """
    K = m0.shape[0]
    S1 = S1 + jitter * jnp.eye(K, dtype=S0.dtype)
    S1 = 0.5 * (S1 + S1.T)
    _, logdet_S1 = jnp.linalg.slogdet(S1)
    _, logdet_S0 = jnp.linalg.slogdet(S0)
    trace_term = jnp.trace(jnp.linalg.solve(S1, S0))
    delta = m1 - m0
    quad = delta @ jnp.linalg.solve(S1, delta)
    return 0.5 * (logdet_S1 - logdet_S0 - K + trace_term + quad)


def compute_kl(t: jnp.array, m: jnp.array, S: jnp.array, fp: Callable[[jnp.array], jnp.array], prior: SDE, params: Dict[str, jnp.array], drift_fn: Callable[[jnp.array], Callable], key: jr.PRNGKey, n_z0: int = 1, jitter: float = 1e-8) -> float:
    """
    Approximate (1/2) E[||G(x,t)^{-1}(fp(x) - fq(x))||^2] with a Monte Carlo average, where x ~ N(m, S)
    """
    # Draw samples
    K = m.shape[0]
    _, _, sqrtS, _ = sym_sqrt_and_invsqrt(S, jitter=jitter)
    zs = jr.normal(key, shape=(n_z0, K), dtype=S.dtype)
    xs = m[None, :] + (sqrtS @ zs.T).T # (n_z0, K)

    # Compute diffusion coefficient
    Gts = vmap(lambda x: prior.diffusion(x, t, params["sde_params"]))(xs) # (n_z0, K, K)
    def fq_eval(G, x):
        return drift_fn(G)(x)
    fqxs = vmap(fq_eval)(Gts, xs)
    div_GGts = vmap(lambda x: prior.div_GGt(x, t, params["sde_params"]))(xs)

    norms = jnp.sum(jnp.square((vmap(lambda x, fqx, Gt, div_GGt: jnp.linalg.solve(Gt, fp(x) - fqx - 0.5 * div_GGt))(xs, fqxs, Gts, div_GGts))), axis=-1) # (n_z0), ||fp(x) - fq(x)||^2
    return 0.5 * norms.mean() # ()


# --------------------- Exact likelihood ---------------------
def reconstruction_loss_exact(key: jr.PRNGKey, ctx_seq: jnp.array, process_ctx: Callable[[jnp.array, jnp.array], jnp.array], params: Dict[str, jnp.array], post_net: Posterior, ys: jnp.array, obs_times: jnp.array, likelihood: Likelihood, mask: Optional[jnp.array] = None) -> float:
    def _eval_ell(keyt, yt, t):
        t_arr = jnp.array([t], dtype=obs_times.dtype)
        mt, Rt = post_net.apply(params["posterior_params"], t_arr, ctx_seq, process_ctx)
        return (-1.0) * likelihood.ell(yt, t_arr, mt, Rt @ Rt.T, keyt, params["output_params"])

    ells = vmap(_eval_ell)(jr.split(key, ys.shape[0]), ys, obs_times)

    if mask is None:
        return ells.sum()
    else:
        return (ells * mask).sum()
