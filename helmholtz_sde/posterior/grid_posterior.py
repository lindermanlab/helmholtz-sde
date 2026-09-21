"""
Non-amortized grid-based posterior parameterization for latent SDE models
"""

import jax.numpy as jnp
from flax import linen as nn

from typing import Callable, Optional, Tuple

from helmholtz_sde.posterior.nn_posterior import make_cholesky
from helmholtz_sde.posterior.posterior import Posterior


class GridInferenceNetwork(Posterior):
    """
    Non-amortized posterior storing m(t_i) and R(t_i) at fixed grid points
    Returns linearly interpolated values between grid points so that t -> (m(t), R(t)) is continuous
    """
    K: int # latent dimension
    grid_times: jnp.array # (N) sorted time grid, not learned
    m_grid_init: Optional[jnp.array] = None # optional warm-start for mean grid
    raw_R_grid_init: Optional[jnp.array] = None # optional warm-start for covariance grid
    jitter: float = 1e-8 # jitter

    @nn.compact
    def __call__(self, t: jnp.array, ctx: jnp.array, process_ctx: Callable[[jnp.array, jnp.array], jnp.array]) -> Tuple[jnp.array, jnp.array]:
        N = self.grid_times.shape[0]
        n_sym = self.K * (self.K + 1) // 2

        if self.m_grid_init is not None:
            m_grid = self.param("m_grid", lambda _key, _shape: self.m_grid_init, (N, self.K))
        else:
            m_grid = self.param("m_grid", nn.initializers.zeros, (N, self.K))
        if self.raw_R_grid_init is not None:
            raw_R_grid = self.param("raw_R_grid", lambda _key, _shape: self.raw_R_grid_init, (N, n_sym))
        else:
            raw_R_grid = self.param("raw_R_grid", nn.initializers.zeros, (N, n_sym))

        # Left grid index and interpolation weight
        idx = jnp.searchsorted(self.grid_times, t[0], side='right') - 1
        idx = jnp.clip(idx, 0, N - 2)
        h = self.grid_times[idx + 1] - self.grid_times[idx]
        # Clamp the weight so that m and R are extrapolated as constants outside the grid
        w = jnp.clip((t[0] - self.grid_times[idx]) / h, 0.0, 1.0)

        # Linearly interpolate m
        m = (1 - w) * m_grid[idx] + w * m_grid[idx + 1]

        # Linearly interpolate the Cholesky factor R, which stays lower triangular with a positive diagonal
        R_left = make_cholesky(raw_R_grid[idx], self.K, jitter=self.jitter)
        R_right = make_cholesky(raw_R_grid[idx + 1], self.K, jitter=self.jitter)
        R = (1 - w) * R_left + w * R_right
        return m, R
