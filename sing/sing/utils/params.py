"""
Defines different parameterizations of a linear, Gaussian Markov chain
"""
import jax
import jax.numpy as jnp
from typing import TypedDict

class NaturalParams(TypedDict):
    """
    The natural parameters of a linear, Gaussian Markov chain
    
    - J: a shape (T, D, D) array, the D x D diagonal blocks of the precision matrix
    - L: a shape (T-1, D, D) array, the D x D lower-diagonal blocks of the precision matrix
    - h: a shape (T, D) array, the precision-weighted mean
    """
    J: jnp.array
    L: jnp.array
    h: jnp.array

class MeanParams(TypedDict):
    """
    The mean parameters of a linear, Gaussian Markov chain

    - Ex: a shape (T, D) array, the marginal means E[x_i]
    - ExxT: a shape (T, D, D) array, the second moments E[x_i x_i^T]
    - ExxnT: a shape (T-1, D, D) array, the cross second moments E[x_i x_{i+1}^T]
    """
    Ex: jnp.array
    ExxT: jnp.array
    ExxnT: jnp.array

class MarginalParams(TypedDict):
    """
    The pairwise marginal parameters of a linear, Gaussian Markov chain

    - m: a shape (T, D) array, the marginal means E[x_i]
    - S: a shape (T, D, D) array, the marginal covariances Var(x_i)
    - SS: a shape (T-1, D, D) array, the covariance between consecutive states x_i and x_{i+1}, Cov(x_i, x_{i+1})
    """
    m: jnp.array
    S: jnp.array
    SS: jnp.array