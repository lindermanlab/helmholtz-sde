"""
Primary module for divergence-free Helmholtz corrections to the posterior drift
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from helmholtz_sde.utils.general_helpers import sym_sqrt_and_invsqrt
from helmholtz_sde.utils.hermite import HermiteCoeffs, evaluate_field, hermite_to_monomial, project_divfree, whiten_frame

from typing import Callable, NamedTuple, Optional, Tuple

# --------------------- Base classes ---------------------
@dataclass(frozen=True)
class HelmCorrection(ABC):
    """
    Base class for divergence-free corrections h to the posterior reference drift

    Given the residual r(x) = fp(x) - fq(x) between the prior drift fp and the posterior reference drift fq,
    and the posterior marginal q = N(m, S), a correction produces a polynomial vector field h of degree ell
    satisfying div(q * h) = 0, so that the corrected posterior drift fq + h has the same marginals as fq

    NOTE: HelmCorrection assumes identity diffusion coefficient; whitening is performed by _whiten
    """
    ell: int # polynomial degree of the correction field
    jitter: float = 1e-8

    def __post_init__(self) -> None:
        if not isinstance(self.ell, int) or self.ell < 0:
            raise ValueError(f"ell must be a non-negative int, got {self.ell}")

    @abstractmethod
    def fit(self, m: jnp.array, S: jnp.array, fp: Callable[[jnp.array], jnp.array], fq: Callable[[jnp.array], jnp.array], key: Optional[jr.PRNGKey] = None) -> NamedTuple:
        """
        Performs all state-independent work and returns the coefficients of h as a NamedTuple
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, coeffs: NamedTuple, x: jnp.array) -> jnp.array:
        """
        Evaluates the correction h(x) at a state x (K)
        """
        raise NotImplementedError

    @abstractmethod
    def monomial_coeffs(self, coeffs: NamedTuple) -> Tuple[jnp.array, ...]:
        """
        Coefficients of h in the monomial basis of the original coordinates, highest degree first
        h_i(x) = sum_{d=0}^{ell} P[d][i, j_1, ..., j_d] x_{j_1} ... x_{j_d} is returned as (P[ell], ..., P[0])
        """
        raise NotImplementedError


@dataclass(frozen=True)
class HermiteCorrection(HelmCorrection):
    """
    Base class for corrections represented in the normalized, multivariate Hermite basis

    In the whitened frame x = m + S^{1/2} Q v the marginal q is standard normal, so the Hermite products are
    orthonormal and the divergence-free constraint is closed-form

    Subclasses supply only hermite_coeffs, their estimate of the Hermite coefficients of the residual; the whitening,
    the projection, the evaluation and the monomial coefficients are shared
    """
    @abstractmethod
    def hermite_coeffs(self, rho: Callable[[jnp.array], jnp.array], K: int, key: Optional[jr.PRNGKey] = None, dtype: Optional[jnp.dtype] = None) -> jnp.array:
        """
        Hermite coefficients b (Na, K) of the residual rho, expressed in the whitened frame
        """
        raise NotImplementedError

    def fit(self, m: jnp.array, S: jnp.array, fp: Callable[[jnp.array], jnp.array], fq: Callable[[jnp.array], jnp.array], key: Optional[jr.PRNGKey] = None) -> HermiteCoeffs:
        Q, lam, sqrtS, invsqrtS, frame = whiten_frame(S, jitter=self.jitter)

        def rho(v: jnp.array) -> jnp.array:
            x = m + frame @ v
            return Q.T @ (invsqrtS @ (fp(x) - fq(x)))

        b = self.hermite_coeffs(rho, m.shape[0], key=key, dtype=m.dtype)
        c = project_divfree(b, lam, self.ell)
        return HermiteCoeffs(c=c, Q=Q, sqrtS=sqrtS, invsqrtS=invsqrtS, m=m)

    def evaluate(self, coeffs: HermiteCoeffs, x: jnp.array) -> jnp.array:
        return evaluate_field(coeffs, x, self.ell)

    def monomial_coeffs(self, coeffs: HermiteCoeffs) -> Tuple[jnp.array, ...]:
        return hermite_to_monomial(coeffs, self.ell)


# --------------------- KL estimator and corrected drift ---------------------
def _fit_whitened(m: jnp.array, S: jnp.array, fp: Callable[[jnp.array], jnp.array], fq: Callable[[jnp.array], jnp.array], Gt: jnp.array, correction: HelmCorrection, key: Optional[jr.PRNGKey] = None) -> Tuple[jnp.array, Callable[[jnp.array], jnp.array], Callable[[jnp.array], jnp.array], NamedTuple]:
    """
    Whitens the posterior marginal and the drifts by the state-independent diffusion coefficient, y = G^{-1} x, and fits
    the correction in the whitened coordinates; returns G^{-1}, the whitened drifts and the coefficients
    """
    K = m.shape[0]
    Gtinv = jnp.linalg.solve(Gt, jnp.eye(K, dtype=S.dtype)) # G^{-1}

    # Whiten the mean and covariance
    m_tilde = Gtinv @ m
    S_tilde = Gtinv @ S @ Gtinv.T

    # Transform vector fields accordingly
    fp_tilde = lambda y: Gtinv @ fp(Gt @ y)
    fq_tilde = lambda y: Gtinv @ fq(Gt @ y)
    coeffs = correction.fit(m_tilde, S_tilde, fp_tilde, fq_tilde, key=key)
    return Gtinv, fp_tilde, fq_tilde, coeffs


def compute_kl_helmholtz_approx(
    m: jnp.array, # (K) posterior mean
    S: jnp.array, # (K, K) posterior covariance
    fp: Callable[[jnp.array], jnp.array], # prior drift
    fq: Callable[[jnp.array], jnp.array], # posterior reference drift
    Gt: jnp.array, # (K, K) diffusion coefficient; NOTE: assumed to be state-independent
    key: jr.PRNGKey, # random key
    correction: HelmCorrection, # divergence-free correction to the posterior reference drift
    n_z0: int = 1, # number of Monte Carlo samples of the state
) -> float:
    """
    Approximates (1/2) E||G^{-1}(fp(x) - fq(x) - h(x))||^2 with a Monte Carlo average, where x ~ N(m, S) and h is the
    divergence-free correction fitted in the coordinates whitened by G with the key fold_in(key, 1)
    """
    K = m.shape[0]
    _, _, sqrtS, _ = sym_sqrt_and_invsqrt(S, jitter=correction.jitter)
    Gtinv, fp_tilde, fq_tilde, coeffs = _fit_whitened(m, S, fp, fq, Gt, correction, key=jr.fold_in(key, 1)) # the fit uses a key independent of the state batch

    # Draw samples from the posterior, then map to y = G^{-1}x
    zs = jr.normal(key, shape=(n_z0, K), dtype=S.dtype)
    xs = m[None, :] + (sqrtS @ zs.T).T
    ys = xs @ Gtinv.T

    bar_f_y = vmap(lambda y: correction.evaluate(coeffs, y))(ys)
    r_y = vmap(fp_tilde)(ys) - vmap(fq_tilde)(ys)
    kl_hat = 0.5 * jnp.mean(jnp.sum(jnp.square(r_y - bar_f_y), axis=-1))
    return kl_hat


def make_corrected_drift(
    m: jnp.array, # (K) posterior mean
    S: jnp.array, # (K, K) posterior covariance
    fp: Callable[[jnp.array], jnp.array], # prior drift
    fq: Callable[[jnp.array], jnp.array], # posterior reference drift
    Gt: jnp.array, # (K, K) diffusion coefficient; NOTE: assumed to be state-independent
    correction: HelmCorrection, # divergence-free correction to the posterior reference drift
    key: Optional[jr.PRNGKey] = None, # random key, required by Monte Carlo corrections
) -> Callable[[jnp.array], jnp.array]:
    """
    Returns the corrected posterior drift x -> fq(x) + G h(G^{-1} x), where h is the correction fitted in the
    coordinates whitened by G, matching the drift whose KL is estimated by compute_kl_helmholtz_approx (with the same
    key fold_in(key, 1) for Monte Carlo corrections)
    """
    Gtinv, _, _, coeffs = _fit_whitened(m, S, fp, fq, Gt, correction, key=key)

    def fq_star(x: jnp.array) -> jnp.array:
        return fq(x) + Gt @ correction.evaluate(coeffs, Gtinv @ x)
    return fq_star
