"""
Quadrature and Monte Carlo nodes for expectations under a standard normal distribution
Used to estimate the coefficients of the Hermite least-squares Helmholtz correction
"""

import functools

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from typing import Optional, Tuple


@functools.lru_cache(maxsize=None)
def gauss_hermite_nodes(K: int, n_nodes: int) -> Tuple[np.array, np.array]:
    """
    Tensor-product Gauss-Hermite quadrature for N(0, I_K)

    Returns nodes (n_nodes^K, K) and weights (n_nodes^K) summing to one
    NOTE: The rule is exact for all polynomials of total degree <= 2 n_nodes - 1
    """
    x1, w1 = np.polynomial.hermite_e.hermegauss(n_nodes)
    w1 = w1 / w1.sum()
    grids = np.meshgrid(*([x1] * K), indexing="ij")
    nodes = np.stack([g.ravel() for g in grids], axis=-1) # (n_nodes^K, K)
    wgrids = np.meshgrid(*([w1] * K), indexing="ij")
    weights = np.prod(np.stack([g.ravel() for g in wgrids], axis=-1), axis=-1) # (n_nodes^K)
    for arr in (nodes, weights):
        arr.setflags(write=False)
    return nodes, weights


def monte_carlo_nodes(key: jr.PRNGKey, K: int, n_mc: int, dtype: Optional[jnp.dtype] = None) -> Tuple[jnp.array, jnp.array]:
    """
    Monte Carlo nodes for N(0, I_K)

    Returns n_mc standard normal samples (n_mc, K) and uniform weights (n_mc) summing to one
    """
    nodes = jr.normal(key, shape=(n_mc, K), dtype=dtype) # (n_mc, K)
    weights = jnp.full((n_mc,), 1.0 / n_mc, dtype=dtype) # (n_mc)
    return nodes, weights
