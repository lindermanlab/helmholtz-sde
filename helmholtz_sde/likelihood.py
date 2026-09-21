"""
Implements the likelihood model p(y|x) in the latent-SDE model
"""

import jax.numpy as jnp
import jax.random as jr

import tensorflow_probability.substrates.jax.distributions as tfd

from typing import Any, Dict
from abc import ABC, abstractmethod


class Likelihood(ABC):
    """
    The likelihood model p(y | x)
    """
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def ll(self, x: jnp.array, y: jnp.array, t: float, output_params: Dict[str, Any]) -> jnp.array:
        """
        Evaluates the log likelihood p(y | x)
        """
        raise NotImplementedError

    @abstractmethod
    def ell(self, y: jnp.array, t: float, mt: jnp.array, St: jnp.array, key: jr.PRNGKey, output_params: Dict[str, Any]) -> jnp.array:
        """
        Evaluates the expected log likelihood p(y | x)
        """
        raise NotImplementedError


class Gaussian(Likelihood):
    """
    Gaussian likelihood model with linear decoder Cx + d and covariance R
    Supports:
      - R.shape == (D) : diagonal covariance diag(R)
      - R.shape == (D,D) : full covariance
    """
    def _symmetrize_spd(self, R: jnp.array, jitter: float = 1e-8) -> jnp.array:
        assert R.ndim == 2
        R = 0.5 * (R + R.T)
        R = R + jitter * jnp.eye(R.shape[0], dtype=R.dtype)
        return R

    def ll(self, x: jnp.array, y: jnp.array, t: float, output_params: Dict[str, jnp.array]) -> jnp.array:
        C, d, R = output_params["C"], output_params["d"], output_params["R"]
        mean = C @ x + d

        if R.ndim == 1:
            return tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.sqrt(R)).log_prob(y)
        elif R.ndim == 2:
            R = self._symmetrize_spd(R, jitter=1e-8)
            return tfd.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=R).log_prob(y)
        else:
            raise ValueError(f"R must have ndim 1 or 2, got shape {R.shape}")

    def ell(self, y: jnp.array, t: float, mt: jnp.array, St: jnp.array, key: jr.PRNGKey, output_params: Dict[str, Any]) -> jnp.array:

        C, R = output_params["C"], output_params["R"]
        ll = self.ll(mt, y, t, output_params)

        # Covariance contribution from x ~ N(m(t), S(t))
        obs_cov = C @ St @ C.T
        if R.ndim == 1:
            correction = -0.5 * jnp.sum(jnp.diag(obs_cov) / R)
        elif R.ndim == 2:
            R = self._symmetrize_spd(R, jitter=1e-8)
            correction = -0.5 * jnp.trace(jnp.linalg.solve(R, obs_cov))
        else:
            raise ValueError(f"R must have ndim 1 or 2, got shape {R.shape}")
        return ll + correction
