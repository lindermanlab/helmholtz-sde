"""
Taylor approximation to the Helmholtz decomposition

The residual r = fp - fq is Taylor-expanded to order ell about the posterior mean, and the divergence-free part of
that polynomial surrogate is computed in the Hermite basis
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr

from helmholtz_sde.helmholtz.correction import HermiteCorrection
from helmholtz_sde.utils.general_helpers import symmetrize_full
from helmholtz_sde.utils.hermite import monomial_to_hermite

from typing import Callable, Optional


@dataclass(frozen=True)
class TaylorCorrection(HermiteCorrection):
    """
    Polynomial Helmholtz approximation of order ell

    The residual is replaced by its order-ell Taylor polynomial about m, whose coefficients are the derivative
    tensors of the whitened residual at the origin
    """

    def hermite_coeffs(self, rho: Callable[[jnp.array], jnp.array], K: int, key: Optional[jr.PRNGKey] = None, dtype: Optional[jnp.dtype] = None) -> jnp.array:
        del key # the Taylor expansion is deterministic
        v0 = jnp.zeros((K,), dtype=dtype)

        # Derivative tensors of rho at the origin; E[l] has shape (K,) * (l + 1), with axis 0 the output and
        # axes 1..l the derivative axes, so that rho(v) is approximated by sum_l E[l] v^l / l!
        E = [rho(v0)]
        fn = rho
        for l in range(1, self.ell + 1):
            fn = jax.jacrev(fn)
            E.append(symmetrize_full(fn(v0), n_axes=l, start_axis=1))
        return monomial_to_hermite(E, K, self.ell)
