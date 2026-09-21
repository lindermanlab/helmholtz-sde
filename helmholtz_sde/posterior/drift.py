"""
Posterior drift computation
Implements the symmetric and square root reference drifts, and the posterior SDE (reference drift plus divergence-free correction)
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from functools import partial

from helmholtz_sde.sde import SDE
from helmholtz_sde.helmholtz.correction import HelmCorrection, make_corrected_drift
from helmholtz_sde.utils.general_helpers import simulate_sde, sym_sqrt_and_invsqrt
from helmholtz_sde.posterior.posterior import Posterior

from typing import Any, Callable, Dict, Optional, Tuple


# --------------------- Posterior SDE ---------------------
class PosteriorSDE(SDE):
    """
    Approximate posterior SDE of one trial, dx = fq*(x, t) dt + G(x, t) dw, sharing the same diffusion coefficient
    as the prior

    fq* is the reference drift plus (i) the divergence-free correction fitted at each time t to the posterior marginal N(m(t), S(t));
    or (ii) the Ito term (1/2) div(GG^T) when the diffusion coefficient is state-dependent

    NOTE: Helmholtz-SDE does NOT support state-dependent diffusion coefficient
    """
    def __init__(self, prior: SDE, post_net: Posterior, ctx_seq: jnp.array, process_ctx: Callable[[jnp.array, jnp.array], jnp.array], gauge: str = "sqrt", div_free: Optional[HelmCorrection] = None, key: Optional[jr.PRNGKey] = None) -> None:
        super().__init__(prior.K)
        self.prior = prior
        self.post_net = post_net
        self.ctx_seq = ctx_seq # (T, H) encoded observations of the trial
        self.process_ctx = process_ctx # (ctx_seq, t) -> context vector, with the observation times of the trial bound
        self.drift_fn = get_reference_drift_fn(gauge)
        self.div_free = div_free
        self.key = key # required by Monte Carlo corrections

    def _time(self, t: jnp.array) -> jnp.array:
        return jnp.reshape(jnp.asarray(t, dtype=self.ctx_seq.dtype), (1,))

    def marginal(self, t: jnp.array, params: Dict[str, Any]) -> Tuple[jnp.array, jnp.array]:
        """
        Mean m(t) and covariance square root R(t) of the posterior marginal q(x, t)
        """
        return self.post_net.apply(params["posterior_params"], self._time(t), self.ctx_seq, self.process_ctx)

    def drift(self, x: jnp.array, t: jnp.array, params: Dict[str, Any]) -> jnp.array:
        t = self._time(t)
        mt, Rt, dmt, dRt = apply_inference_net_time_derivs(self.post_net, params["posterior_params"], t, self.ctx_seq, self.process_ctx)
        Gt = self.prior.diffusion(x, t, params["sde_params"])
        fq = self.drift_fn(mt, Rt, dmt, dRt, Gt)
        if self.div_free is None:
            return fq(x) + 0.5 * self.prior.div_GGt(x, t, params["sde_params"])
        fp = lambda x_: self.prior(x_, t, params["sde_params"])
        return make_corrected_drift(mt, Rt @ Rt.T, fp, fq, Gt, correction=self.div_free, key=self.key)(x) # fq + G h(G^{-1} x)

    def diffusion(self, x: jnp.array, t: jnp.array, params: Dict[str, Any]) -> jnp.array:
        return self.prior.diffusion(x, self._time(t), params["sde_params"])

    def div_GGt(self, x: jnp.array, t: jnp.array, params: Dict[str, Any]) -> jnp.array:
        return self.prior.div_GGt(x, self._time(t), params["sde_params"])


def simulate_posterior_samples(key: jr.PRNGKey, posterior_sde: PosteriorSDE, params: Dict[str, Any], t_max: float, n_timesteps: int, n_samples: int) -> jnp.array:
    """
    Draw n_samples sample paths (n_samples, n_timesteps + 1, K) from the posterior SDE on [0, t_max], with x(0) ~ q(x, 0)
    """
    key_init, key_paths = jr.split(key, 2)
    m0, R0 = posterior_sde.marginal(0.0, params)
    x0 = m0[None, :] + jr.normal(key_init, shape=(n_samples, posterior_sde.K), dtype=m0.dtype) @ R0.T
    return vmap(partial(simulate_sde, sde=posterior_sde, sde_params=params, t_max=t_max, n_timesteps=n_timesteps))(jr.split(key_paths, n_samples), x0)


# --------------------- Reference drifts ---------------------
def compute_posterior_drift_sqrt(m: jnp.array, R: jnp.array, dm: jnp.array, dR: jnp.array, Gt: jnp.array, symmetrize_R: bool = True) -> Callable[[jnp.array], jnp.array]:
    """
    Forms the posterior drift at time t from (m(t), R(t), d/dt m(t), d/dt R(t)) and
    diffusion coefficient GG^T, using a square root R(t) of the marginal covariance
    S(t) = R(t) R(t)^T

    When symmetrize_R=True, both R and dR are replaced by the unique symmetric PD
    square root of S and its JVP-derived time derivative

    NOTE: DOES support state-dependent G
    """
    if symmetrize_R:
        def _sym_sqrt(R_):
            return sym_sqrt_and_invsqrt(R_ @ R_.T)[2]
        R, dR = jax.jvp(_sym_sqrt, (R,), (dR,))

    def _fq(x):
        v = x - m

        u = jnp.linalg.solve(R, v)
        drift_term = dm + dR @ u

        # Since S = R R^T, S^{-1} = (R^T)^{-1} R^{-1}
        score_term = - jnp.linalg.solve(R.T, u)
        return drift_term + 0.5 * (Gt @ Gt.T) @ score_term
    return _fq


def compute_posterior_drift_sym(m: jnp.array, R: jnp.array, dm: jnp.array, dR: jnp.array, Gt: jnp.array) -> Callable[[jnp.array], jnp.array]:
    """
    Forms the posterior drift at time t from (m(t), R(t), d/dt m(t), d/dt R(t)) and diffusion coefficient GG^T
    using the marginal mean m(t) and the square root R(t) of the marginal covariance S(t)

    Chooses the unique symmetric A that attains the marginals N(m(t), S(t))
    NOTE: DOES NOT support state-dependent G
    """
    S  = R @ R.T
    dS = dR @ R.T + R @ dR.T
    C = dS - Gt @ Gt.T

    A = solve_sylvester(S, C)
    def _fq(z):
        return dm + A @ (z - m)
    return _fq


def solve_sylvester(S: jnp.array, C: jnp.array, jitter: float = 1e-8) -> jnp.array:
    """
    Solves the Sylvester equation
    S B + B S = C
    by computing the eigendecomposition of S

    Assumes S and C are symmetric, S is positive definite
    """
    K = S.shape[0]
    evals, U = jnp.linalg.eigh(S + jitter * jnp.diag(jnp.arange(1, K + 1, dtype=S.dtype)))
    evals = jnp.clip(evals, jitter, 1. / jitter)

    Ct = U.T @ C @ U
    denom = evals[:, None] + evals[None, :]
    Bt = Ct / denom
    B = U @ Bt @ U.T
    return B


def apply_inference_net_time_derivs(model: Posterior, params: Dict, t: jnp.array, ctx_seq: jnp.array, process_ctx: Callable[[jnp.array, jnp.array], jnp.array]) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    """
    Computes m(t), R(t), dm/dt, dR/dt using JVP
    """
    def f(tt):
        return model.apply(params, tt, ctx_seq, process_ctx)
    (mt, Rt), (dmt, dRt) = jax.jvp(f, (t,), (jnp.array([1.0], dtype=t.dtype),))
    return mt, Rt, dmt, dRt


def get_reference_drift_fn(gauge: str) -> Callable[..., Callable[[jnp.array], jnp.array]]:
    """
    Constructor of the posterior reference drift of the gauge: sqrt (SDE Matching) or sym (SVISE)
    """
    if gauge == "sqrt":
        return compute_posterior_drift_sqrt
    if gauge == "sym":
        return compute_posterior_drift_sym
    raise ValueError("Only gauges [sqrt, sym] are supported")
