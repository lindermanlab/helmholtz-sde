import jax
import jax.random as jr
import jax.numpy as jnp
from jax import vmap, lax

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from typing import Callable, Dict, Optional


def simulate_sde(key: jr.PRNGKey, 
                x0: jnp.array, 
                f: Callable[[jnp.array, float], jnp.array], 
                t_max: float, 
                n_timesteps: int, 
                inputs: Optional[jnp.array] = None, 
                input_effect: Optional[jnp.array] = None, 
                sigma: Optional[Callable[[jnp.array, float], jnp.array]] = None) -> jnp.array:
    """
    Function for simulating a general SDE with drift f and diffusion sigma
    NOTE: samples a single path; to sample multiple paths one should use vmap
    
    Params:
    -------------
    key: random key for sampling
    x0: a shape (D) array, the initial condition
    f: drift function, a function of x and t
    t_max: total duration of trial
    n_timesteps: total number of timesteps to simulate
    NOTE: the difference between timesteps is t_max / (n_timesteps - 1)
    inputs: a shape (T, n_inputs) array, inputs in discrete time
    input_effect: a shape (D, n_inputs) array, the linear mapping from inputs to the latent space
    sigma: diffusion function, a function of x and t
    
    Returns:
    -------------
    xs: (T, D) simulated latent SDE path
    """
    
    D = x0.shape[0]
    dt = (t_max - 0.) / (n_timesteps - 1)
    def _step(x, arg):
        key, t, input = arg
        drift = f(x, t).reshape(D)
        diffusion = sigma(x, t).reshape(D, D)
        next_x = tfd.MultivariateNormalFullCovariance(loc = x + drift * dt + input_effect @ input * dt, covariance_matrix = diffusion @ diffusion.T *
                                                      dt).sample(seed=key).astype(jnp.float64)
        return next_x, x
    
    if inputs is None:
        inputs = jnp.zeros((n_timesteps, 1))
        input_effect = jnp.zeros((len(x0), 1))
    
    # Default is unit variance
    if sigma is None:
        sigma = lambda x, t: jnp.eye(D)
    
    keys = jr.split(key, n_timesteps)
    ts = jnp.linspace(0., t_max, n_timesteps)
    _, xs = lax.scan(_step, x0, (keys, ts, inputs))
    return xs

def simulate_gaussian_obs(key: jr.PRNGKey, xs: jnp.array, output_params: Dict[str, jnp.array]) -> jnp.array:
    """
    Simulate Gaussian observations at every timestep of a latent SDE

    Params:
    -------------
    key: a random key for sampling
    xs: a shape (T, D) array, the latent SDE path
    output_params: a dictionary representing the parameters of the likelihood, containing
        - C: a shape (N, D) array, the linear part of the output mapping
        - d: a shape (N) array, the vector offset part of the output mapping
        - R: a shape (N) array, the diagonal entries of the covariance

    Returns:
    -------------
    gaussian_obs: (T, N) noisy observations
    """
    C, d, R = output_params['C'], output_params['d'], output_params['R']
    means = jnp.einsum('dk,tk->td', C, xs) + d 
    gaussian_obs = tfd.Normal(loc=means, scale=jnp.sqrt(R)).sample(seed=key)
    return gaussian_obs

def simulate_poisson_obs(dt: float, key: jr.PRNGKey, xs: jnp.array, output_params: Dict[str, jnp.array], link: Optional[Callable[[float], float]] = None, include_dt: bool = False) -> jnp.array:
    """
    Simulate Poisson observations from a latent SDE.
    y|x ~ Pois(inv_link(Cx + d)*dt)

    Params:
    ---------------
    dt: time discretization at which to simulate observations
    key: jr.PRNGKey
    xs: a shape (T, D) array, the latent SDE path
    output_params: a dictionary representing the parameters of the likelihood, containing
        - C: a shape (N, D) array, the linear part of the output mapping
        - d: a shape (N) array, the vector offset part of the output mapping
    link: inverse link function
        - If None, defaults to 'exp' canonical link
    include_dt: a boolean, whether to scale the Poisson rate by dt
    
    Returns:
    ---------------
    poisson_obs: (T, N) Poisson counts
    """
    if link is None:
        link = jnp.exp

    C, d = output_params['C'], output_params['d']
    activations = jnp.einsum('dk,tk->td', C, xs) + d
    rate = link(activations) 
    if include_dt:
        rate *= dt
    poisson_obs = tfd.Poisson(rate=rate).sample(seed=key) # (n_timesteps, K)
    return poisson_obs

def simulate_generalized_poisson_obs(obs_dim: int, key: jr.PRNGKey, xs: jnp.array, link: Optional[Callable[[float, int], float]] = None) -> jnp.array:
    """
    Simulate Poisson observations with generalized inverse link functions from a latent SDE.
    y|x ~ Pois(f_n(x)), where inverse link f_n can vary across output dimensions n.

    Params:
    ---------------
    dt: time discretization at which to simulate observations
    key: jr.PRNGKey
    xs: a shape (T, D) array, the latent SDE path
    link: inverse link function of the form, link(x, [obs_dim]) -> rate
        - If None, defaults to 'exp' canonical link per observation dimension
    
    Returns:
    ---------------
    poisson_obs: (T, N) Poisson counts
    """
    if link is None:
        link = lambda x, idx: jnp.exp(x)

    rates = vmap(vmap(link, (None, 0)), (0, None))(xs, jnp.arange(0, obs_dim)) # (n_timesteps, obs_dim)
    poisson_obs = tfd.Poisson(rate=rates).sample(seed=key)
    return poisson_obs
