"""
Normalized multivariate Hermite basis for the divergence-free Helmholtz corrections
"""

import functools
import math

import jax.numpy as jnp
import numpy as np

from helmholtz_sde.utils.general_helpers import sym_sqrt_and_invsqrt

from typing import Dict, List, NamedTuple, Tuple


# --------------------- Multi-index tables ---------------------
def _multi_indices_upto(dims: int, degree: int) -> List[Tuple[int, ...]]:
    """
    All multi-indices over dims coordinates of total degree at most degree
    """
    rows = []
    row = [0] * dims
    def visit(axis: int, remaining: int):
        if axis == dims:
            rows.append(tuple(row))
            return
        for v in range(remaining + 1):
            row[axis] = v
            visit(axis + 1, remaining - v)
        row[axis] = 0
    visit(0, degree)
    return rows


@functools.lru_cache(maxsize=None)
def index_tables(K: int, ell: int) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Static multi-index tables for the Hermite basis and the divergence-free projection

    alpha_table (Na, K) indexes the basis functions of degree <= ell, beta_table (Nb, K) the constraints of degree
    1 <= |beta| <= ell + 1, and the two index arrays map between them by lowering or raising one coordinate
    """
    alphas = _multi_indices_upto(K, ell)
    betas = _multi_indices_upto(K, ell + 1)[1:] # drop the all-zero tuple
    alpha_index = {a: i for i, a in enumerate(alphas)}
    beta_index = {b: i for i, b in enumerate(betas)}

    alpha_table = np.array(alphas, dtype=np.int32) # (Na, K)
    beta_table = np.array(betas, dtype=np.int32) # (Nb, K)

    beta_minus_idx = np.zeros((len(betas), K), dtype=np.int32)
    for bi, b in enumerate(betas):
        for i in range(K):
            if b[i] > 0:
                a = list(b)
                a[i] -= 1
                beta_minus_idx[bi, i] = alpha_index[tuple(a)]

    alpha_plus_idx = np.zeros((len(alphas), K), dtype=np.int32)
    for ai, a in enumerate(alphas):
        for i in range(K):
            b = list(a)
            b[i] += 1
            alpha_plus_idx[ai, i] = beta_index[tuple(b)]

    for arr in (alpha_table, beta_table, beta_minus_idx, alpha_plus_idx):
        arr.setflags(write=False)
    return alpha_table, beta_table, beta_minus_idx, alpha_plus_idx


# --------------------- Whitened frame ---------------------
class HermiteCoeffs(NamedTuple):
    """
    Coefficients and whitened frame of a divergence-free correction

    NOTE: c holds normalized-Hermite coefficients in the coordinates x = m + sqrtS Q v;
    use evaluate_field to evaluate the correction and hermite_to_monomial for its polynomial coefficients
    """
    c: jnp.array # (Na, K) Hermite coefficients of the correction in v-coords
    Q: jnp.array # (K, K) eigenbasis of S
    sqrtS: jnp.array # (K, K) symmetric square root of S
    invsqrtS: jnp.array # (K, K) inverse symmetric square root of S
    m: jnp.array # (K) posterior mean


def whiten_frame(S: jnp.array, jitter: float = 1e-8) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]:
    """
    Returns (Q, lam, sqrtS, invsqrtS, frame) for the marginal covariance S

    The frame satisfies x = m + frame v with v standard normal under the marginal, and lam holds the
    eigenvalues of S^{-1}, which weight the divergence-free projection
    """
    Q, evals_S, sqrtS, invsqrtS = sym_sqrt_and_invsqrt(S, jitter=jitter)
    lam = 1.0 / evals_S
    frame = sqrtS @ Q
    return Q, lam, sqrtS, invsqrtS, frame


# --------------------- Basis evaluation ---------------------
def hermite_products(v: jnp.array, alpha_table: np.array, max_deg: int) -> jnp.array:
    """
    Evaluate the normalized Hermite products H_alpha(v) for all rows of alpha_table

    Uses the normalized recurrence He_{d+1} = (v He_d - sqrt(d) He_{d-1}) / sqrt(d + 1)
    """
    K = v.shape[0]
    U = [jnp.ones((K,), dtype=v.dtype)]
    if max_deg >= 1:
        U.append(v)
    for d in range(1, max_deg):
        U.append((v * U[d] - jnp.sqrt(float(d)) * U[d - 1]) / jnp.sqrt(float(d + 1)))
    U = jnp.stack(U, axis=0) # (max_deg + 1, K)
    vals = U[alpha_table, np.arange(K)[None, :]] # (Na, K)
    return jnp.prod(vals, axis=1) # (Na)


def evaluate_field(coeffs: HermiteCoeffs, x: jnp.array, ell: int) -> jnp.array:
    """
    Evaluate the correction h(x) = S^{1/2} Q nu(Q^T S^{-1/2}(x - m)) at a state x
    """
    K = coeffs.m.shape[0]
    alpha_table, *_ = index_tables(K, ell)
    v = coeffs.Q.T @ (coeffs.invsqrtS @ (x - coeffs.m))
    nu = coeffs.c.T @ hermite_products(v, alpha_table, ell) # (K)
    return coeffs.sqrtS @ (coeffs.Q @ nu)


# --------------------- Divergence-free projection ---------------------
def project_divfree(b: jnp.array, lam: jnp.array, ell: int) -> jnp.array:
    """
    Project Hermite coefficients b (Na, K) onto the q-divergence-free subspace

    Since (d_i - v_i) H_alpha = -sqrt(alpha_i + 1) H_{alpha + e_i}, the constraint decouples into one scalar
    equation per beta and the lamda-weighted projection is closed-form
    """
    K = lam.shape[0]
    alpha_table, beta_table, beta_minus_idx, alpha_plus_idx = index_tables(K, ell)
    beta_f = jnp.asarray(beta_table, dtype=b.dtype)
    sqrt_beta = jnp.sqrt(beta_f) # (Nb, K), zero where beta_i = 0 so sentinel rows drop out

    # Multipliers mu_beta = sum_i sqrt(beta_i) b[beta - e_i, i] / sum_i lam_i beta_i
    # NOTE: every beta has |beta| >= 1, so the denominator is at least min(lam) > 0
    b_gather = b[beta_minus_idx, np.arange(K)[None, :]] # (Nb, K)
    mu = jnp.sum(sqrt_beta * b_gather, axis=1) / (beta_f @ lam) # (Nb)

    # Projected coefficients c[alpha, i] = b[alpha, i] - lam_i sqrt(alpha_i + 1) mu[alpha + e_i]
    sqrt_alpha_p1 = jnp.sqrt(jnp.asarray(alpha_table, dtype=b.dtype) + 1.0) # (Na, K)
    c = b - lam[None, :] * sqrt_alpha_p1 * mu[alpha_plus_idx] # (Na, K)

    # The constant coefficient vanishes analytically
    c = c.at[0].set(0.0)
    return c


# --------------------- Tensor helpers ---------------------
def _rotate_out_and_deriv(T: jnp.array, M_out: jnp.array, M_deriv: jnp.array) -> jnp.array:
    """
    Contract axis 0 with M_out's first axis and axes 1..rank-1 with M_deriv's first axis
    Used to push polynomial coefficients from one coordinate frame to another
    """
    rank = T.ndim
    if rank == 0:
        return T
    operands = [T, list(range(rank))]
    operands.extend([M_out, [0, rank + 0]])
    for axis in range(1, rank):
        operands.extend([M_deriv, [axis, rank + axis]])
    operands.append(list(range(rank, 2 * rank)))
    return jnp.einsum(*operands)


def _contract_last_axes(T: jnp.array, v: jnp.array, n: int) -> jnp.array:
    """
    Contract the last n axes of T with n copies of vector v
    """
    out = T
    for _ in range(n):
        out = out @ v
    return out


def _trace_last_pairs(T: jnp.array, n: int) -> jnp.array:
    """
    Contract the last 2n axes of T in pairs; T is symmetric in its derivative axes, so the choice of pairs is immaterial
    """
    out = T
    for _ in range(n):
        out = jnp.trace(out, axis1=-2, axis2=-1)
    return out


# --------------------- Monomial tensors <-> Hermite coefficients ---------------------
@functools.lru_cache(maxsize=None)
def _gather_tables(K: int, ell: int) -> Dict[int, Tuple[np.array, Tuple[np.array, ...], np.array]]:
    """
    Per degree d: the rows of alpha_table with |alpha| = d, the expanded coordinate indices of each such alpha,
    and the factor 1 / sqrt(alpha!) relating a symmetric monomial tensor entry to its Hermite coefficient
    """
    alpha_table, *_ = index_tables(K, ell)
    out = {}
    for d in range(ell + 1):
        rows = [ai for ai, a in enumerate(np.asarray(alpha_table)) if a.sum() == d]
        expanded = [np.repeat(np.arange(K), np.asarray(alpha_table)[ai]) for ai in rows]
        factors = [1.0 / math.sqrt(np.prod([math.factorial(int(c)) for c in np.asarray(alpha_table)[ai]])) for ai in rows]
        idx = tuple(np.array([e[axis] for e in expanded], dtype=np.int32) for axis in range(d))
        out[d] = (np.array(rows, dtype=np.int32), idx, np.array(factors, dtype=np.float64))
    return out


@functools.lru_cache(maxsize=None)
def _scatter_tables(K: int, ell: int) -> Dict[int, Tuple[np.array, np.array]]:
    """
    Per degree d: for every index tuple in [K]^d, the row of alpha_table it belongs to and the factor sqrt(alpha!)
    This is the inverse of _gather_tables, filling a dense symmetric tensor from the Hermite coefficients
    """
    alpha_table, *_ = index_tables(K, ell)
    alpha_index = {tuple(int(c) for c in a): ai for ai, a in enumerate(np.asarray(alpha_table))}
    out = {}
    for d in range(ell + 1):
        rows = np.zeros((K,) * d, dtype=np.int32)
        facs = np.zeros((K,) * d, dtype=np.float64)
        for idx in np.ndindex(*((K,) * d)):
            counts = [0] * K
            for j in idx:
                counts[j] += 1
            rows[idx] = alpha_index[tuple(counts)]
            facs[idx] = math.sqrt(np.prod([math.factorial(c) for c in counts]))
        out[d] = (rows, facs)
    return out


def monomial_to_hermite(E: List[jnp.array], K: int, ell: int) -> jnp.array:
    """
    Hermite coefficients of the polynomial p_i(v) = sum_{l<=ell} (1/l!) E[l][i, j_1..j_l] v_{j_1}..v_{j_l}

    Uses b[alpha, i] = E[H_alpha(v) p_i(v)] = (1/sqrt(alpha!)) E[d^alpha p_i(v)] and the fact that
    the Gaussian expectation of a symmetric monomial of degree 2j contributes j traces with weight 1 / (2^j j!)
    """
    dtype = E[0].dtype
    # Trace corrections: B[d] = sum_j tr^j(E[d + 2j]) / (2^j j!)
    B = []
    for d in range(ell + 1):
        acc = jnp.zeros((K,) * (d + 1), dtype=dtype)
        for j in range(0, (ell - d) // 2 + 1):
            acc = acc + _trace_last_pairs(E[d + 2 * j], j) / (2.0 ** j * math.factorial(j))
        B.append(acc)

    # Read each Hermite coefficient off the corresponding entry of the symmetric tensor
    alpha_table, *_ = index_tables(K, ell)
    b = jnp.zeros((alpha_table.shape[0], K), dtype=dtype)
    for d, (rows, idx, factors) in _gather_tables(K, ell).items():
        vals = B[d][None, :] if d == 0 else B[d][(slice(None), *idx)].T # (G_d, K)
        b = b.at[rows].set(jnp.asarray(factors, dtype=dtype)[:, None] * vals)
    return b


def hermite_to_monomial(coeffs: HermiteCoeffs, ell: int) -> Tuple[jnp.array, ...]:
    """
    Monomial coefficients of the correction in the original coordinates, highest degree first

    h_i(x) = sum_{d=0}^{ell} P[d][i, q_1..q_d] x_{q_1}..x_{q_d} is returned as (P[ell], ..., P[0])
    """
    K = coeffs.m.shape[0]
    dtype = coeffs.c.dtype

    # Fill dense symmetric tensors B[d] from the Hermite coefficients
    B = []
    for d, (rows, facs) in _scatter_tables(K, ell).items():
        B.append(jnp.moveaxis(coeffs.c[rows], -1, 0) * jnp.asarray(facs, dtype=dtype))

    # Invert the trace corrections: E[d] = sum_j (-1)^j tr^j(B[d + 2j]) / (2^j j!)
    E = []
    for d in range(ell + 1):
        acc = jnp.zeros((K,) * (d + 1), dtype=dtype)
        for j in range(0, (ell - d) // 2 + 1):
            acc = acc + ((-1.0) ** j) * _trace_last_pairs(B[d + 2 * j], j) / (2.0 ** j * math.factorial(j))
        E.append(acc)

    # Push the coefficients from v-coords to delta = x - m coords, using h(x) = frame nu(invframe delta)
    frame = coeffs.sqrtS @ coeffs.Q
    invframe = coeffs.Q.T @ coeffs.invsqrtS
    P_delta = [_rotate_out_and_deriv(E[l], frame.T, invframe) / math.factorial(l) for l in range(ell + 1)]

    # Shift delta -> x by the binomial expansion of (x - m)
    P_x = []
    for d in range(ell + 1):
        acc = jnp.zeros_like(P_delta[d])
        for l in range(d, ell + 1):
            sign = 1.0 if ((l - d) % 2 == 0) else -1.0
            acc = acc + math.comb(l, d) * sign * _contract_last_axes(P_delta[l], coeffs.m, l - d)
        P_x.append(acc)
    return tuple(P_x[d] for d in range(ell, -1, -1))
