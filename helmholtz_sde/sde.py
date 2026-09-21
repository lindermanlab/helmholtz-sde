"""
Implements the (continuous-time) prior p(x) in the latent-SDE model
"""

import jax
import jax.numpy as jnp

from flax import linen as nn

from typing import Any, Callable, Dict, Optional
from abc import ABC, abstractmethod


# --------------------- SDE classes ---------------------
class SDE(ABC):
    """
    Base stochastic differential equation (SDE) class
    dx(t) = f(x(t), t)dt + G(x(t), t)dw(t)
    """
    def __init__(self, K: int) -> None:
        self.K = K
        super().__init__()

    @abstractmethod
    def drift(self, x: jnp.array, t: jnp.array, sde_params: Dict[str, Any]) -> jnp.array:
        """
        Drift function
        """
        raise NotImplementedError

    def diffusion(self, x: jnp.array, t: jnp.array, sde_params: Dict[str, Any]) -> jnp.array:
        """
        Diffusion coefficient
        """
        if "G" in sde_params:
            return sde_params["G"]
        else:
            return jnp.eye(self.K)

    def div_GGt(self, x: jnp.array, t: float, sde_params: Dict[str, Any]) -> jnp.array:
        """
        Row-wise divergence of the diffusion covariance GG^T
        """
        return jnp.zeros((self.K))

    def __call__(self, x: jnp.array, t: jnp.array, sde_params: Dict[str, Any]) -> jnp.array:
        return self.drift(x, t, sde_params)


class LinearSDE(SDE):
    """
    A time-homogeneous linear SDE
    dx(t) = {Ax(t) + b} dt + Gdw(t)
    """
    def drift(self, x: jnp.array, t: jnp.array, sde_params: Dict[str, Any]) -> jnp.array:
        A, b = sde_params["A"], sde_params["b"]
        return A @ x + b

class NeuralSDE(SDE):
    """
    A prior SDE with neural network drift and diffusion
    """
    def __init__(self, K: int, apply_fn_drift: Callable[[Dict[str, Any], jnp.array, jnp.array], jnp.array], apply_fn_diffusion: Optional[Callable[[Dict[str, Any], jnp.array, jnp.array], jnp.array]] = None, div_GGt_apply_fn: Optional[Callable[[Dict[str, Any], jnp.array, jnp.array], jnp.array]] = None) -> None:
        super().__init__(K)
        self.drift_apply_fn = apply_fn_drift
        if apply_fn_diffusion is None:
            self.diffusion_apply_fn = lambda *args: jnp.eye(self.K)
            self.div_GGt_apply_fn = lambda *args: jnp.zeros((self.K))
        else:
            self.diffusion_apply_fn = apply_fn_diffusion
            if div_GGt_apply_fn is None:
                def _default_div_GGt_apply_fn(params, x, t):
                    def GGt_fn(z):
                        G = self.diffusion_apply_fn(params, z, t)
                        return G @ G.T
                    J = jax.jacfwd(GGt_fn)(x)
                    return jnp.einsum("ijj->i", J)
                self.div_GGt_apply_fn = _default_div_GGt_apply_fn
            else:
                self.div_GGt_apply_fn = div_GGt_apply_fn

    def drift(self, x: jnp.array, t: float, sde_params: Dict[str, Any]) -> jnp.array:
        return self.drift_apply_fn(sde_params['network_params_drift'], x, t)

    def diffusion(self, x: jnp.array, t: float, sde_params: Dict[str, Any]) -> jnp.array:
        return self.diffusion_apply_fn(sde_params.get('network_params_diffusion'), x, t)

    def div_GGt(self, x: jnp.array, t: float, sde_params: Dict[str, Any]) -> jnp.array:
        return self.div_GGt_apply_fn(sde_params.get('network_params_diffusion'), x, t)

# --------------------- Neural networks for NeuralSDE ---------------------
class DriftNetwork(nn.Module):
    """
    Neural network representing the SDE drift, with softplus activations by default
    """
    hidden_dim: int
    K: int
    depth: int = 2
    include_time: bool = False
    activation: Callable[[jnp.array], jnp.array] = nn.softplus

    @nn.compact
    def __call__(self, x: jnp.array, t: jnp.array) -> jnp.array:
        if self.include_time:
            xx = jnp.concatenate([x, t], axis=-1)
        else:
            xx = x
        for _ in range(self.depth):
            xx = self.activation(nn.Dense(self.hidden_dim)(xx))
        out = nn.Dense(self.K)(xx)
        return out


class ScalarVolNet(nn.Module):
    """
    Neural network representing a single diagonal entry of the diffusion coefficient, with softplus activations

    Takes as input the corresponding latent dimension
    """
    hidden_dim: int
    include_time: bool = False

    @nn.compact
    def __call__(self, x_i: jnp.array, t: jnp.array) -> jnp.array:
        x_i = jnp.asarray(x_i)
        inp = x_i[None]

        if self.include_time:
            inp = jnp.concatenate([inp, t], axis=-1)
        z = nn.Dense(self.hidden_dim)(inp)
        z = nn.softplus(z)
        z = nn.Dense(1)(z)
        sigma_i = nn.sigmoid(z)[0]
        return sigma_i


class DiffusionNetwork(nn.Module):
    """
    Neural network representing a diagonal diffusion coefficient

    The ith diagonal entry depends only on the ith latent dimension
    """
    hidden_dim: int
    K: int
    depth: int = 1
    include_time: bool = False

    def setup(self):
        self.vol_nets = [ScalarVolNet(hidden_dim=self.hidden_dim, include_time=self.include_time, name=f"vol_net_{i}") for i in range(self.K)]

    def sigma(self, x: jnp.array, t: jnp.array) -> jnp.array:
        return jnp.stack([self.vol_nets[i](x[i], t) for i in range(self.K)], axis=-1)

    def __call__(self, x: jnp.array, t: jnp.array) -> jnp.array:
        sigma = self.sigma(x, t)
        return jnp.diag(sigma)

    def div_GGt(self, x: jnp.array, t: jnp.array) -> jnp.array:
        div_terms = []
        for i in range(self.K):
            f_i = lambda x_i: self.vol_nets[i](x_i, t) ** 2
            div_terms.append(jax.grad(f_i)(x[i]))
        return jnp.stack(div_terms, axis=-1)
