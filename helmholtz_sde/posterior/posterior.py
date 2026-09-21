"""
Base class for posterior q(x) in the latent SDE model
"""

from abc import abstractmethod

import jax.numpy as jnp
from flax import linen as nn

from typing import Callable, Tuple


class Posterior(nn.Module):
    """
    All posteriors map (t, ctx, process_ctx) -> (m(t), R(t)) where m is a
    mean vector and R is a sqrt-covariance factor satisfying S(t) = R(t) R(t)^T
    """
    @abstractmethod
    def __call__(
        self,
        t: jnp.array, # time at which to evaluate the posterior
        ctx: jnp.array, # context tensor (T+1, H)
        process_ctx: Callable[[jnp.array, jnp.array], jnp.array], # function for processing the context vector
    ) -> Tuple[jnp.array, jnp.array]:
        raise NotImplementedError
