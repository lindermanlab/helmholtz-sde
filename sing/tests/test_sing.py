"""
Test code for SING
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu" # run tests on CPU backend

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap
from jax import random as jr
from jax import lax

from sing.sde import LinearSDE
from sing.likelihoods import Gaussian, Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.utils.params import *
from sing.utils.sing_helpers import dynamics_to_natural_params, natural_to_mean_params, natural_to_marginal_params, mean_to_marginal_params
from sing.sing import compute_elbo_over_batch, fit_variational_em, nat_grad_likelihood, nat_grad_transition, sing_update

import io
import contextlib
import pytest
from typing import Dict, Tuple
from functools import partial
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

# --------------------- Fixtures ---------------------
@pytest.fixture(scope="session")
def n_trials(): return 3

@pytest.fixture(scope="session")
def n_trials_large(): return 30

@pytest.fixture(scope="session")
def latent_dim(): return 2

@pytest.fixture(scope="session")
def obs_dim(): return 10

@pytest.fixture(scope="session")
def n_timesteps(): return 101

@pytest.fixture(scope="session")
def t_max(): return 1.0

@pytest.fixture(scope="session")
def t_grid(t_max, n_timesteps): return jnp.linspace(0., t_max, n_timesteps)

def generate_trial_mask(n_trials, n_timesteps): return jnp.ones((n_trials, n_timesteps)).astype(bool)

@pytest.fixture(scope="session")
def trial_mask(n_trials, n_timesteps): return generate_trial_mask(n_trials, n_timesteps)

@pytest.fixture(scope="session")
def trial_mask_large(n_trials_large, n_timesteps): return generate_trial_mask(n_trials_large, n_timesteps)

def generate_t_mask(n_trials, n_timesteps, key, zero_frac=0.7):
    n_zeros = int(zero_frac * n_timesteps)
    def make_mask(k):
        perm = jax.random.permutation(k, n_timesteps)
        mask = jnp.ones(n_timesteps)
        mask = mask.at[perm[:n_zeros]].set(0)
        return mask
    keys = jax.random.split(key(), n_trials)
    masks = jax.vmap(make_mask)(keys)
    return masks

@pytest.fixture(scope="function")
def t_mask(n_trials, n_timesteps, key): return generate_t_mask(n_trials, n_timesteps, key)

@pytest.fixture(scope="function")
def t_mask_large(n_trials_large, n_timesteps, key): return generate_t_mask(n_trials_large, n_timesteps, key, zero_frac=0.)

@pytest.fixture(scope="function")
def key():
    # A generator-style fixture that yields a new key each time
    master = jr.PRNGKey(0)
    def next_key():
        nonlocal master
        master, subkey = jr.split(master)
        return subkey
    return next_key

@pytest.fixture(scope="function")
def linear_dynamics(latent_dim, n_timesteps, t_max):
    dt = t_max / (n_timesteps - 1)
    theta = jnp.pi / 250.0
    rot_scale = 0.997
    rot = rot_scale * jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                                 [jnp.sin(theta),  jnp.cos(theta)]])
    drift_params_true = {
        "A": (rot - jnp.eye(latent_dim)) / dt,
        "b": jnp.zeros((latent_dim,))
    }
    return dt, drift_params_true

@pytest.fixture(scope="function")
def output_params_true(latent_dim, obs_dim, key):
    kC, kd, kR = jr.split(key(), 3)
    C = jr.normal(kC, (obs_dim, latent_dim))
    d = jr.normal(kd, (obs_dim,))
    R = 0.1 * jnp.ones((obs_dim,))
    return {"C": C, "d": d, "R": R}

@pytest.fixture(scope="function")
def linear_prior(linear_dynamics, key):
    dt, drift_params_true = linear_dynamics
    latent_dim = drift_params_true['A'].shape[0]
    splits = jr.split(key(), 2)
    sigma = 1.
    sde_params = {
        'A': drift_params_true['A'] + sigma * jr.normal(splits[0], (latent_dim, latent_dim)),
        'b': drift_params_true['b'] + sigma * jr.normal(splits[1], (latent_dim))
    }
    return LinearSDE(latent_dim), sde_params

def generate_samples_lds(latent_dim, n_timesteps, n_trials, t_max, linear_dynamics, output_params_true, key):
    dt, drift_params_true = linear_dynamics
    def drift_fn(x, t): return drift_params_true["A"] @ x + drift_params_true["b"]

    splits = jr.split(key(), 3)
    x0 = jr.uniform(splits[0], (n_trials, latent_dim), minval=-2.0, maxval=2.0)
    xs = vmap(partial(simulate_sde, f=drift_fn, t_max=t_max, n_timesteps=n_timesteps))(jnp.array(jr.split(splits[1], n_trials)), x0)  # (n_trials, T, D)
    ys = vmap(partial(simulate_gaussian_obs, output_params=output_params_true))(jnp.array(jr.split(splits[2], n_trials)), xs) # (n_trials, T, N)
    return xs, ys

@pytest.fixture(scope="function")
def lds_samples(latent_dim, n_timesteps, n_trials, t_max, linear_dynamics, output_params_true, key):
    return generate_samples_lds(latent_dim, n_timesteps, n_trials, t_max, linear_dynamics, output_params_true, key)

@pytest.fixture(scope="function")
def lds_samples_large(latent_dim, n_timesteps, n_trials_large, t_max, linear_dynamics, output_params_true, key):
    return generate_samples_lds(latent_dim, n_timesteps, n_trials_large, t_max, linear_dynamics, output_params_true, key)

def generate_init_params(lds_samples, latent_dim):
    xs, _ = lds_samples
    n_trials, _, _ = xs.shape
    return {'mu0': xs[:, 0], 'V0': jnp.eye(latent_dim)[None, :, :].repeat(n_trials, axis=0)}

@pytest.fixture(scope="function")
def init_params(lds_samples, latent_dim):
    return generate_init_params(lds_samples, latent_dim)

@pytest.fixture(scope="function")
def init_params_large(lds_samples_large, latent_dim):
    return generate_init_params(lds_samples_large, latent_dim)

@pytest.fixture(scope="function")
def likelihood(lds_samples, t_mask):
    _, ys = lds_samples
    return Gaussian(ys, t_mask)

# --------------------- Helper functions ---------------------
def dynamics_to_mean_params(A, b, Q) -> MeanParams:
    T, D = b.shape
    m0, S0 = b[0], Q[0]

    def step(carry, inputs):
        m_t, S_t = carry
        A_t, b_t, Q_t = inputs
        # Predict next marginal
        m_next = A_t @ m_t + b_t
        S_next = A_t @ S_t @ A_t.T + Q_t
        # Cross covariance
        SS_t = A_t @ S_t
        return (m_next, S_next), (m_next, S_next, SS_t)

    (_, _), (m_rest, S_rest, SS) = lax.scan(step, (m0, S0), (A, b[1:], Q[1:]))

    # Stack full sequences
    m = jnp.vstack([m0[None, :], m_rest]) # (T, D)
    S = jnp.concatenate([S0[None, ...], S_rest], 0) # (T, D, D)

    # Build ExxT and ExxnT
    outer = vmap(lambda x1, x2: jnp.outer(x1, x2))
    Ex = m
    ExxT = S + outer(m, m)
    ExxnT = SS + outer(m[1:], m[:-1])
    return {'Ex': Ex, 'ExxT': ExxT, 'ExxnT': ExxnT}

def linear_SDE_to_dynamics_params(A, b, Q, xs, dt, sigma = 1.):
    D = A.shape[0]
    n_trials, n_timesteps, D = xs.shape
    Atilde, btilde, Qtilde = jnp.eye(D) + dt * A, dt * b, dt * Q 
    Atilde_batched, btilde_batched, Qtilde_batched = jnp.broadcast_to(Atilde, (n_trials, n_timesteps - 1, D, D)), jnp.concatenate([xs[:, 0:1, :], jnp.broadcast_to(btilde, (n_trials, n_timesteps - 1, D))], axis=1),  jnp.concatenate([(sigma**2) * jnp.broadcast_to(jnp.eye(D), (n_trials, 1, D, D)), jnp.broadcast_to(Qtilde, (n_trials, n_timesteps - 1, D, D))], axis=1)
    return Atilde_batched, btilde_batched, Qtilde_batched

# --------------------- Tests -------------------
def test_natural_to_mean_conversion(latent_dim, n_timesteps, key):
    """
    Test SING conversion between natural -> mean parameters
    """

    As = jr.normal(key(), (n_timesteps-1, latent_dim, latent_dim))
    bs = jr.normal(key(), (n_timesteps, latent_dim))
    Qs = jnp.stack([jnp.eye(latent_dim)] * n_timesteps, axis=0)
    trial_mask=jnp.ones((n_timesteps)).astype(bool)

    natural_params = dynamics_to_natural_params(As, bs, Qs, trial_mask)
    mean_params, _ = natural_to_mean_params(natural_params, trial_mask, jitter = 0.) # set jitter = 0

    mean_params_true = dynamics_to_mean_params(As, bs, Qs)
    assert jnp.allclose(mean_params["Ex"], mean_params_true["Ex"])
    assert jnp.allclose(mean_params["ExxT"], mean_params_true["ExxT"])
    assert jnp.allclose(mean_params["ExxnT"], mean_params_true["ExxnT"])

def test_natural_grad_likelihood(linear_dynamics, output_params_true, lds_samples, t_mask, likelihood):
    """
    Test the SING natural gradient computation for the log-likelihood term
    """
    dt, drift_params_true = linear_dynamics
    C, d, R = output_params_true["C"], output_params_true["d"], output_params_true["R"]
    xs, ys = lds_samples
    n_trials, T, D = xs.shape

    # Instantiate mean parameters, likelihood 
    As_batched, bs_batched, Qs_batched = linear_SDE_to_dynamics_params(drift_params_true["A"], drift_params_true["b"], jnp.eye(D), xs, dt)
    mean_params = vmap(dynamics_to_mean_params)(As_batched, bs_batched, Qs_batched)

    # Analytical natural gradient
    J_true = -jnp.broadcast_to((0.5 * C.T @ jnp.diag(1.0 / R) @ C), (n_trials, T, D, D)) * t_mask[..., None, None]
    h_true = (ys - d) @ ((1.0 / R)[:, None] * C) * t_mask[..., None]
    nat_grad_true = {'J': J_true, 'h': h_true}

    # Compute natural gradient 
    nat_grad = vmap(partial(nat_grad_likelihood, likelihood=likelihood, output_params=output_params_true))(mean_params, t_mask, ys)

    assert jnp.allclose(nat_grad['J'], nat_grad_true['J'])
    assert jnp.allclose(nat_grad['h'], nat_grad_true['h'])

def test_natural_grad_transition(linear_dynamics, output_params_true, lds_samples, linear_prior, init_params, t_grid, trial_mask, key):
    """
    Test the SING natural gradient computation for the transition term 
    """
    dt, drift_params_true = linear_dynamics
    C, d, R = output_params_true["C"], output_params_true["d"], output_params_true["R"]
    xs, ys = lds_samples
    n_trials, T, D = xs.shape

    # Instantiate a linear SDE prior
    linear_sde, prior_params = linear_prior

    # Analytical natural gradient
    As_batched, bs_batched, Qs_batched = linear_SDE_to_dynamics_params(prior_params["A"], prior_params["b"], jnp.eye(D), xs, dt)
    nat_grad_true = vmap(dynamics_to_natural_params)(As_batched, bs_batched, Qs_batched, trial_mask)
    
    # Compute natural gradient 
    As_batched, bs_batched, Qs_batched = linear_SDE_to_dynamics_params(drift_params_true["A"], drift_params_true["b"], jnp.eye(D), xs, dt)
    mean_params = vmap(dynamics_to_mean_params)(As_batched, bs_batched, Qs_batched) # mean parameters of the variational posterior
    nat_grad = vmap(partial(nat_grad_transition, key(), linear_sde, None, prior_params, input_effect=jnp.zeros((D, 1)), sigma=1.))(trial_mask, init_params, t_grid[None, :].repeat(n_trials, axis=0), mean_params, jnp.zeros((n_trials, T, 1)))

    assert jnp.allclose(nat_grad['J'], nat_grad_true['J'])
    assert jnp.allclose(nat_grad['L'], nat_grad_true['L'])
    assert jnp.allclose(nat_grad['h'], nat_grad_true['h'])

def test_ELBO_computation(linear_dynamics, output_params_true, lds_samples, likelihood, linear_prior, init_params, t_mask, trial_mask, t_grid, key):
    """
    Test the SING ELBO computation
    """
    dt, drift_params_true = linear_dynamics
    C, d, R = output_params_true["C"], output_params_true["d"], output_params_true["R"]
    xs, ys = lds_samples
    n_trials, T, D = xs.shape
    linear_sde, prior_params = linear_prior

    # Compute natural and marginal params of the variational posterior
    As_batched, bs_batched, Qs_batched = linear_SDE_to_dynamics_params(drift_params_true["A"], drift_params_true["b"], jnp.eye(D), xs, dt)
    natural_params = vmap(dynamics_to_natural_params)(As_batched, bs_batched, Qs_batched, trial_mask)
    mean_params, Ap = vmap(partial(natural_to_mean_params, jitter = 0.))(natural_params, trial_mask)
    marginal_params = vmap(mean_to_marginal_params)(mean_params)

    # Compute the ELBO with SING
    elbo, _, kl_term_true, _ = compute_elbo_over_batch(key(), ys, t_mask, trial_mask, linear_sde, likelihood, t_grid, prior_params, init_params, output_params_true, natural_params, marginal_params, Ap, jnp.zeros((n_trials, T, 1)), jnp.zeros((D, 1)), 1.)

    # Compute the true ELBO
    ## Compute the negative KL term (as a Bregman divergence)
    As_batched, bs_batched, Qs_batched = linear_SDE_to_dynamics_params(prior_params["A"], prior_params["b"], jnp.eye(D), xs, dt)
    natural_params_prior = vmap(dynamics_to_natural_params)(As_batched, bs_batched, Qs_batched, trial_mask) # natural parameters and log normalizer of the prior
    _, Ap_prior = vmap(partial(natural_to_mean_params, jitter = 0.))(natural_params_prior, trial_mask) 
    def _inner_product(mu_q, eta_p, eta_q):
        dJ, dL, dh = (eta_p['J'] - eta_q['J']) * trial_mask[..., None, None], (eta_p['L'] - eta_q['L']) * trial_mask[..., :-1, None, None], (eta_p['h'] - eta_q['h']) * trial_mask[..., None]  
        mat_inner, vec_inner = lambda A, B: jnp.trace(A.T @ B), lambda a, b: jnp.dot(a, b)  # matrix and vector inner products
        bt_mat_inner,  bt_vec_inner = jax.vmap(jax.vmap(mat_inner)), jax.vmap(jax.vmap(vec_inner)) # vmap over batch and time
        return bt_mat_inner(dJ, mu_q['ExxT']).sum() + bt_mat_inner(dL, mu_q['ExxnT']).sum() + bt_vec_inner(dh, mu_q['Ex']).sum()
    prod = _inner_product(mean_params, natural_params_prior, natural_params).sum()
    kl_term = Ap_prior.sum() - Ap.sum() - prod 

    ## Compute the expected LL term (same as in SING ELBO computation)
    ell_term = vmap(partial(likelihood.ell_over_time, output_params=output_params_true))(ys, {'m': marginal_params['m'], 'S': marginal_params['S']}, t_mask).sum()

    elbo_true = ell_term - kl_term
    assert jnp.isclose(elbo, elbo_true)
    assert jnp.isclose(kl_term, kl_term_true)

def _check_linear_inference(linear_dynamics, output_params_true, lds_samples, likelihood, linear_prior, init_params, t_mask, trial_mask, t_grid, key):
    dt, drift_params_true = linear_dynamics
    C, d, R = output_params_true["C"], output_params_true["d"], output_params_true["R"]
    xs, ys = lds_samples
    n_trials, T, D = xs.shape
    linear_sde, prior_params = linear_prior

    # Compute true posterior
    As_batched, bs_batched, Qs_batched = linear_SDE_to_dynamics_params(prior_params["A"], prior_params["b"], jnp.eye(D), xs, dt)
    natural_params_prior = vmap(dynamics_to_natural_params)(As_batched, bs_batched, Qs_batched, trial_mask)
    natural_params_true = {
        'J': -jnp.broadcast_to((0.5 * C.T @ jnp.diag(1.0 / R) @ C), (n_trials, T, D, D)) * t_mask[..., None, None] + natural_params_prior['J'], 
        'L': natural_params_prior['L'],
        'h': (ys - d) @ ((1.0 / R)[:, None] * C) * t_mask[..., None] + natural_params_prior['h']
    }

    # Perform a single SING step with rho = 1
    rho, n_steps = 1., 1 # use update parameter \rho = 1.
    As_batched, bs_batched, Qs_batched = linear_SDE_to_dynamics_params(drift_params_true["A"], drift_params_true["b"], jnp.eye(D), xs, dt) 
    natural_params = vmap(dynamics_to_natural_params)(As_batched, bs_batched, Qs_batched, trial_mask) # natural parameters of variational posterior
    mean_params = vmap(dynamics_to_mean_params)(As_batched, bs_batched, Qs_batched)
    natural_params = vmap(partial(sing_update, linear_sde, likelihood, t_grid, gp_post=None, drift_params=prior_params, output_params=output_params_true, input_effect=jnp.zeros((D, 1)), sigma=1., rho=rho, n_iters=n_steps, jitter=0.))(jnp.array(jr.split(key(), n_trials)), ys, t_mask, trial_mask, natural_params, init_params, jnp.zeros((n_trials, T, 1)))

    assert jnp.allclose(natural_params['J'], natural_params_true['J'])
    assert jnp.allclose(natural_params['L'], natural_params_true['L'])
    assert jnp.allclose(natural_params['h'], natural_params_true['h'])

def test_linear_inference(linear_dynamics, output_params_true, lds_samples, likelihood, linear_prior, init_params, t_mask, trial_mask, t_grid, key):
    """
    Test that SING performs exact inference in the linear, Gaussian setting
    """
    _check_linear_inference(linear_dynamics, output_params_true, lds_samples, likelihood, linear_prior, init_params, t_mask, trial_mask, t_grid, key)

def test_inference_variable_length_trials(linear_dynamics, output_params_true, lds_samples, likelihood, linear_prior, init_params, t_mask, t_grid, key):
    """
    Test that SING performs exact inference for variable length trials
    """
    xs, _ = lds_samples
    n_trials, T, _ = xs.shape
    cutoffs = jax.random.randint(key(), (n_trials,), 0, T + 1) # randomly truncate each trial
    idx = jnp.arange(T)[None, :]    
    trial_mask  = idx < cutoffs[:, None]
    _check_linear_inference(linear_dynamics, output_params_true, lds_samples, likelihood, linear_prior, init_params, t_mask, trial_mask, t_grid, key)

def test_linear_learning(linear_dynamics, output_params_true, lds_samples_large, linear_prior, init_params_large, t_mask_large, trial_mask_large, t_grid, key):
    """
    Test the M-step for learning A, b in a linear SDE prior
    Checks that the error in A, b decreases after learning 
    """
    dt, drift_params_true = linear_dynamics
    C, d, R = output_params_true["C"], output_params_true["d"], output_params_true["R"]
    xs, ys = lds_samples_large
    n_trials, T, D = xs.shape
    linear_sde, prior_params = linear_prior
    init_params, t_mask, trial_mask = init_params_large, t_mask_large, trial_mask_large
    likelihood = Gaussian(ys, t_mask)

    n_iters = 100
    n_iters_m = 20 # Each M-step consists of 20 SGD updates
    rho_sched = jnp.ones((n_iters))
    learning_rate = jnp.logspace(-1, -3, n_iters)

    # Run vEM algorithm (w/ 1 SING step per iter)
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        output = fit_variational_em(key(), linear_sde, likelihood, t_grid, prior_params, init_params, output_params_true, trial_mask=trial_mask, batch_size=None, rho_sched=rho_sched, learning_rate=learning_rate, n_iters=n_iters, n_iters_e=1, learn_output_params=False)
    drift_params_learned = output[3]

    assert jnp.linalg.norm(drift_params_learned['A'] - drift_params_true['A']) < jnp.linalg.norm(prior_params['A'] - drift_params_true['A'])
    assert jnp.linalg.norm(drift_params_learned['b'] - drift_params_true['b']) < jnp.linalg.norm(prior_params['b'] - drift_params_true['b'])