"""
Neural network posterior parameterizations for latent SDE models; can be amortized across trials
"""

import jax.numpy as jnp
from flax import linen as nn

from typing import Callable

from helmholtz_sde.posterior.posterior import Posterior

# --------------------- Inference network ---------------------
def fill_strictly_lower(offdiag_params: jnp.array, K: int) -> jnp.array:
    idx = jnp.tril_indices(K, -1)
    L = jnp.zeros(offdiag_params.shape[:-1] + (K, K), dtype=offdiag_params.dtype)
    L = L.at[..., idx[0], idx[1]].set(offdiag_params)
    return L


def make_cholesky(raw_params: jnp.array, K: int, jitter: float = 1e-8) -> jnp.array:
    """
    Cholesky factor R (..., K, K) of S = R R^T from K(K + 1) / 2 unconstrained parameters
    """
    raw_diag = raw_params[..., :K]
    raw_offdiag = raw_params[..., K:]
    L = fill_strictly_lower(raw_offdiag, K)
    d = nn.softplus(raw_diag) + jitter
    return L + jnp.eye(K, dtype=raw_params.dtype) * d[..., None, :]


class InferenceNetwork(Posterior):
    """
    Neural network parameterization of the posterior, parameterizes a dense covariance matrix
    """
    hidden_dim: int
    K: int
    depth: int = 2
    jitter: float = 1e-8

    @nn.compact
    def __call__(self, t: jnp.array, ctx: jnp.array, process_ctx: Callable[[jnp.array, jnp.array], jnp.array]):
        n_sym = self.K * (self.K + 1) // 2

        ctx_vec = process_ctx(ctx, t)
        z = jnp.concatenate([ctx_vec, t], axis=-1)
        for _ in range(self.depth):
            z = nn.silu(nn.Dense(self.hidden_dim)(z))
        out = nn.Dense(self.K + n_sym)(z)

        m = out[..., :self.K]
        raw_R = out[..., self.K:]
        R = make_cholesky(raw_R, self.K, jitter=self.jitter)
        return m, R


class InferenceNetworkDiagonal(Posterior):
    """
    Same as InferenceNetwork, but uses a diagonal covariance
    """
    hidden_dim: int
    K: int
    depth: int = 2
    jitter: float = 1e-8

    @nn.compact
    def __call__(self, t: jnp.array, ctx: jnp.array, process_ctx: Callable[[jnp.array, jnp.array], jnp.array]):

        ctx_vec = process_ctx(ctx, t)
        z = jnp.concatenate([ctx_vec, t], axis=-1)
        for _ in range(self.depth):
            z = nn.silu(nn.Dense(self.hidden_dim)(z))
        out = nn.Dense(2 * self.K)(z)

        m = out[..., :self.K]
        log_sigma = out[..., self.K:]

        sigma = jnp.exp(log_sigma) + self.jitter
        R = jnp.einsum("...i,ij->...ij", sigma, jnp.eye(self.K, dtype=out.dtype))
        return m, R
