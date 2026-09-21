"""
Projection onto divergence-free polynomial vector fields

The residual r = fp - fq is projected, in the L2(q) norm, onto the subspace of q-divergence-free polynomial vector
fields of degree at most ell

Unlike the Taylor approximation, which reads the residual off its derivatives at the posterior mean, the Hermite coefficients
here are estimated by integration against the marginal
"""

import functools
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import vmap

from helmholtz_sde.helmholtz.correction import HermiteCorrection
from helmholtz_sde.utils.general_helpers import symmetrize_full
from helmholtz_sde.utils.hermite import hermite_products, index_tables
from helmholtz_sde.utils.quadrature import gauss_hermite_nodes, monte_carlo_nodes

from typing import Callable, Dict, List, Optional, Tuple


# --------------------- Stein index tables ---------------------
@functools.lru_cache(maxsize=None)
def _stein_index_tables(K: int, ell: int, s: int) -> Dict[int, Tuple[np.array, np.array, np.array, np.array]]:
    """
    Static tables for the scalar-order Stein estimator of the Hermite coefficients

    For each alpha the shift kappa_alpha peels min(s, |alpha|) units off alpha one at a time from the largest
    remaining entry (ties broken by lowest coordinate index)

    Rows are grouped by d = |kappa_alpha|; group d maps to (rows, herm_rows, factors, deriv_axes) where rows indexes
    alpha_table, herm_rows is the row of alpha - kappa_alpha, factors is sqrt((alpha - kappa_alpha)! / alpha!), and
    deriv_axes is a (d, G_d) array of the sorted kappa coordinates
    """
    alpha_table, *_ = index_tables(K, ell)
    alphas = [tuple(int(v) for v in row) for row in np.asarray(alpha_table)]
    alpha_index = {a: i for i, a in enumerate(alphas)}

    grouped: Dict[int, List[Tuple[int, int, float, Tuple[int, ...]]]] = {}
    for ai, a in enumerate(alphas):
        rem = list(a)
        coords = []
        for _ in range(min(s, sum(a))):
            j = int(np.argmax(rem)) # ties resolve to the lowest index
            rem[j] -= 1
            coords.append(j)
        # Falling factorial alpha! / (alpha - kappa)! as an exact integer
        ratio = 1
        for j in range(K):
            for step in range(a[j] - rem[j]):
                ratio *= (a[j] - step)
        factor = 1.0 / math.sqrt(ratio)
        entry = (ai, alpha_index[tuple(rem)], factor, tuple(sorted(coords)))
        grouped.setdefault(len(coords), []).append(entry)

    out = {}
    for d, entries in grouped.items():
        rows = np.array([e[0] for e in entries], dtype=np.int32)
        herm_rows = np.array([e[1] for e in entries], dtype=np.int32)
        factors = np.array([e[2] for e in entries], dtype=np.float64)
        deriv_axes = np.array([e[3] for e in entries], dtype=np.int32).reshape(len(entries), d).T # (d, G_d)
        for arr in (rows, herm_rows, factors, deriv_axes):
            arr.setflags(write=False)
        out[d] = (rows, herm_rows, factors, deriv_axes)
    return out


# --------------------- Least-squares correction ---------------------
@dataclass(frozen=True)
class LeastSquaresCorrection(HermiteCorrection):
    """
    Least-squares projection of the residual r = fp - fq onto the subspace of q-divergence-free polynomial vector
    fields of degree <= ell, optimal in the L2(q) norm over that subspace

    The Hermite coefficients of r are estimated by Monte Carlo with n_mc samples, or by tensor-product Gauss-Hermite
    quadrature with n_nodes points per dimension when n_nodes is set; quadrature is exact for integrands of total degree
    <= 2 n_nodes - 1 and makes the correction deterministic given (m, S)

    kappa > 0 activates the Stein integration-by-parts estimator of order min(kappa, ell): each coefficient trades
    min(kappa, |alpha|) units of Hermite degree (peeled from the largest entries of alpha) for derivatives of the
    residual, costing one extra order of automatic differentiation per unit
    """
    kappa: int = 0 # Stein estimator order
    n_mc: int = 64 # number of Monte Carlo samples used to estimate the Hermite coefficients
    n_nodes: Optional[int] = None # Gauss-Hermite nodes per dimension; if None, uses Monte Carlo with n_mc samples

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.kappa, int) or self.kappa < 0:
            raise ValueError(f"kappa must be a non-negative int, got {self.kappa}")
        if self.n_nodes is None and self.n_mc < 1:
            raise ValueError(f"n_mc must be positive, got {self.n_mc}")
        if self.n_nodes is not None and self.n_nodes < 1:
            raise ValueError(f"n_nodes must be positive, got {self.n_nodes}")

    def hermite_coeffs(self, rho: Callable[[jnp.array], jnp.array], K: int, key: Optional[jr.PRNGKey] = None, dtype: Optional[jnp.dtype] = None) -> jnp.array:
        if self.n_nodes is None and key is None:
            raise ValueError("the Monte Carlo estimator requires a PRNG key; pass key or set n_nodes for quadrature")
        ell = self.ell

        # Nodes and weights: Monte Carlo samples or Gauss-Hermite quadrature
        if self.n_nodes is None:
            vs, w = monte_carlo_nodes(key, K, self.n_mc, dtype=dtype) # (N, K), (N)
        else:
            nodes, weights = gauss_hermite_nodes(K, self.n_nodes)
            vs, w = jnp.asarray(nodes, dtype=dtype), jnp.asarray(weights, dtype=dtype) # (N, K), (N)

        rhos = vmap(rho)(vs) # (N, K)
        alpha_table, *_ = index_tables(K, ell)
        H = vmap(lambda v_: hermite_products(v_, alpha_table, ell))(vs) # (N, Na)

        s_eff = min(self.kappa, ell) # shifts are at most |alpha|
        if s_eff == 0:
            return jnp.einsum("n,na,ni->ai", w, H, rhos) # (Na, K)

        # Stein estimator: b[alpha, i] = factor * sum_n w_n d^kappa_alpha rho_i(v_n) H_{alpha - kappa_alpha}(v_n)
        stein_groups = _stein_index_tables(K, ell, s_eff)
        b = jnp.zeros((alpha_table.shape[0], K), dtype=rhos.dtype)
        # NOTE: the jacrev + gather pattern materializes the full K^{d+1} derivative tensor per sample, which is a mild overcompute at d >= 2
        fn = rho
        for d in range(0, s_eff + 1):
            if d > 0:
                fn = jax.jacrev(fn)
                Dd = vmap(fn)(vs) # (N, K) + (K,) * d
                Dd = symmetrize_full(Dd, n_axes=d, start_axis=2)
            rows, herm_rows, factors, deriv_axes = stein_groups[d]
            if d == 0:
                b = b.at[rows].set(jnp.einsum("n,ng,ni->gi", w, H[:, rows], rhos))
            else:
                # Adjacent 1-D advanced indices broadcast elementwise, giving (N, K, G_d)
                sel = Dd[(slice(None), slice(None), *deriv_axes)]
                vals = jnp.einsum("n,nig,ng->gi", w, sel, H[:, herm_rows])
                b = b.at[rows].set(jnp.asarray(factors, dtype=vals.dtype)[:, None] * vals)
        return b


# Default correction used by train
DEFAULT_CORRECTION = LeastSquaresCorrection(ell=1, kappa=1, n_mc=1)
