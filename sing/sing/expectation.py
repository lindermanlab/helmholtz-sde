import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky 
import jax.random as jr
from jax import vmap
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
from numpy.polynomial.hermite_e import hermegauss

from typing import Callable, Tuple

from abc import ABC, abstractmethod

class Expectation(ABC):
    """
    An object for integration with respect to a D-dimensional Gaussian distribution
    """
    
    def __init__(self, D: int) -> None:
        """
        Params:
        -------------
        D: The dimension of the input space
        """
        self.D = D
        super().__init__()

    @abstractmethod
    def gaussian_int(self, fn: Callable[jnp.array, jnp.array], m: jnp.array, S: jnp.array, **kwargs) -> jnp.array:
        """
        Approximates E[f(x)], x ~ N(m, S)

        Params:
        -------------
        fn: a function R^D -> R^{D'}
        m: the mean of x
        S: the covariance of x

        Returns:
        -------------
        Efn: an approximation to the expectation of f(x) under x ~ N(m, S) 
        """
        raise NotImplementedError

class GaussHermiteQuadrature(Expectation):
    def __init__(self, D: int, n_quad: int = 10) -> None:
        """
        Params:
        -------------
        n_quad: number of quadrature points per dimension, default 10; 
        the total number of quadrature points is n_quad^D

        NOTE: GaussHermiteQuadrature is only recommended for low-dimensional (1D, 2D, 3D) input spaces
        """
        super().__init__(D)
        self.weights, self.unit_sigmas = self.compute_weights_and_sigmas(D, n_quad)

    def compute_weights_and_sigmas(self, D: int, n_quad: int) -> Tuple[jnp.array, jnp.array]:
        """
        Computes weights and sigma-points for Gauss-Hermite quadrature
        """
        samples_1d, weights_1d = jnp.array(hermegauss(n_quad)) # weights are proportional to standard Gaussian density
        weights_1d /= weights_1d.sum() # normalize weights 
        weights_rep = [weights_1d for _ in range(D)]
        samples_rep = [samples_1d for _ in range(D)]
        weights = jnp.stack(jnp.meshgrid(*weights_rep), axis=-1).reshape(-1, D).prod(axis=1)
        unit_sigmas = jnp.stack(jnp.meshgrid(*samples_rep), axis=-1).reshape(-1, D)
        return weights, unit_sigmas
    
    def gaussian_int(self, fn: Callable[jnp.array, jnp.array], m: jnp.array, S: jnp.array, **kwargs) -> jnp.array:
        sigmas = m + (jnp.linalg.cholesky(S) @ self.unit_sigmas[...,None]).squeeze(-1)
        return jnp.tensordot(self.weights, vmap(fn)(sigmas), axes=(0, 0)) # same shape as output of f

class MonteCarlo(Expectation):
    def __init__(self, D: int, N: int = 1) -> None:
        """
        Params:
        -------------
        N: the number of Monte Carlo samples to draw when approximating the expectation, default 1
        """
        super().__init__(D)
        self.N = N
    
    def gaussian_int(self, key: jr.PRNGKey, fn: Callable[jnp.array, jnp.array], m: jnp.array, S: jnp.array) -> jnp.array:
        z = tfd.Normal(loc = 0., scale=1.).sample(sample_shape=(self.N, self.D), seed=key)
        L = cholesky(S, lower=True)
        samps = m[None,:] + z @ L.T # (N, D) # reparameterization trick
        return jnp.mean(vmap(fn)(samps), axis=0) # (D) 